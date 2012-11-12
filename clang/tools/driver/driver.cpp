//===-- driver.cpp - Clang GCC-Compatible Driver --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the entry point to the clang driver; it is a thin wrapper
// for functionality in the Driver clang library.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/OptTable.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Frontend/Utils.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/system_error.h"
#include <cctype>
using namespace clang;
using namespace clang::driver;

llvm::sys::Path GetExecutablePath(const char *Argv0, bool CanonicalPrefixes) {
  if (!CanonicalPrefixes)
    return llvm::sys::Path(Argv0);

  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::Path::GetMainExecutable(Argv0, P);
}

static const char *SaveStringInSet(std::set<std::string> &SavedStrings,
                                   StringRef S) {
  return SavedStrings.insert(S).first->c_str();
}

/// ApplyQAOverride - Apply a list of edits to the input argument lists.
///
/// The input string is a space separate list of edits to perform,
/// they are applied in order to the input argument lists. Edits
/// should be one of the following forms:
///
///  '#': Silence information about the changes to the command line arguments.
///
///  '^': Add FOO as a new argument at the beginning of the command line.
///
///  '+': Add FOO as a new argument at the end of the command line.
///
///  's/XXX/YYY/': Substitute the regular expression XXX with YYY in the command
///  line.
///
///  'xOPTION': Removes all instances of the literal argument OPTION.
///
///  'XOPTION': Removes all instances of the literal argument OPTION,
///  and the following argument.
///
///  'Ox': Removes all flags matching 'O' or 'O[sz0-9]' and adds 'Ox'
///  at the end of the command line.
///
/// \param OS - The stream to write edit information to.
/// \param Args - The vector of command line arguments.
/// \param Edit - The override command to perform.
/// \param SavedStrings - Set to use for storing string representations.
static void ApplyOneQAOverride(raw_ostream &OS,
                               SmallVectorImpl<const char*> &Args,
                               StringRef Edit,
                               std::set<std::string> &SavedStrings) {
  // This does not need to be efficient.

  if (Edit[0] == '^') {
    const char *Str =
      SaveStringInSet(SavedStrings, Edit.substr(1));
    OS << "### Adding argument " << Str << " at beginning\n";
    Args.insert(Args.begin() + 1, Str);
  } else if (Edit[0] == '+') {
    const char *Str =
      SaveStringInSet(SavedStrings, Edit.substr(1));
    OS << "### Adding argument " << Str << " at end\n";
    Args.push_back(Str);
  } else if (Edit[0] == 's' && Edit[1] == '/' && Edit.endswith("/") &&
             Edit.slice(2, Edit.size()-1).find('/') != StringRef::npos) {
    StringRef MatchPattern = Edit.substr(2).split('/').first;
    StringRef ReplPattern = Edit.substr(2).split('/').second;
    ReplPattern = ReplPattern.slice(0, ReplPattern.size()-1);

    for (unsigned i = 1, e = Args.size(); i != e; ++i) {
      std::string Repl = llvm::Regex(MatchPattern).sub(ReplPattern, Args[i]);

      if (Repl != Args[i]) {
        OS << "### Replacing '" << Args[i] << "' with '" << Repl << "'\n";
        Args[i] = SaveStringInSet(SavedStrings, Repl);
      }
    }
  } else if (Edit[0] == 'x' || Edit[0] == 'X') {
    std::string Option = Edit.substr(1, std::string::npos);
    for (unsigned i = 1; i < Args.size();) {
      if (Option == Args[i]) {
        OS << "### Deleting argument " << Args[i] << '\n';
        Args.erase(Args.begin() + i);
        if (Edit[0] == 'X') {
          if (i < Args.size()) {
            OS << "### Deleting argument " << Args[i] << '\n';
            Args.erase(Args.begin() + i);
          } else
            OS << "### Invalid X edit, end of command line!\n";
        }
      } else
        ++i;
    }
  } else if (Edit[0] == 'O') {
    for (unsigned i = 1; i < Args.size();) {
      const char *A = Args[i];
      if (A[0] == '-' && A[1] == 'O' &&
          (A[2] == '\0' ||
           (A[3] == '\0' && (A[2] == 's' || A[2] == 'z' ||
                             ('0' <= A[2] && A[2] <= '9'))))) {
        OS << "### Deleting argument " << Args[i] << '\n';
        Args.erase(Args.begin() + i);
      } else
        ++i;
    }
    OS << "### Adding argument " << Edit << " at end\n";
    Args.push_back(SaveStringInSet(SavedStrings, '-' + Edit.str()));
  } else {
    OS << "### Unrecognized edit: " << Edit << "\n";
  }
}

/// ApplyQAOverride - Apply a comma separate list of edits to the
/// input argument lists. See ApplyOneQAOverride.
static void ApplyQAOverride(SmallVectorImpl<const char*> &Args,
                            const char *OverrideStr,
                            std::set<std::string> &SavedStrings) {
  raw_ostream *OS = &llvm::errs();

  if (OverrideStr[0] == '#') {
    ++OverrideStr;
    OS = &llvm::nulls();
  }

  *OS << "### QA_OVERRIDE_GCC3_OPTIONS: " << OverrideStr << "\n";

  // This does not need to be efficient.

  const char *S = OverrideStr;
  while (*S) {
    const char *End = ::strchr(S, ' ');
    if (!End)
      End = S + strlen(S);
    if (End != S)
      ApplyOneQAOverride(*OS, Args, std::string(S, End), SavedStrings);
    S = End;
    if (*S != '\0')
      ++S;
  }
}

extern int cc1_main(const char **ArgBegin, const char **ArgEnd,
                    const char *Argv0, void *MainAddr);
extern int cc1as_main(const char **ArgBegin, const char **ArgEnd,
                      const char *Argv0, void *MainAddr);

static void ExpandArgsFromBuf(const char *Arg,
                              SmallVectorImpl<const char*> &ArgVector,
                              std::set<std::string> &SavedStrings) {
  const char *FName = Arg + 1;
  OwningPtr<llvm::MemoryBuffer> MemBuf;
  if (llvm::MemoryBuffer::getFile(FName, MemBuf)) {
    ArgVector.push_back(SaveStringInSet(SavedStrings, Arg));
    return;
  }

  const char *Buf = MemBuf->getBufferStart();
  char InQuote = ' ';
  std::string CurArg;

  for (const char *P = Buf; ; ++P) {
    if (*P == '\0' || (isspace(*P) && InQuote == ' ')) {
      if (!CurArg.empty()) {

        if (CurArg[0] != '@') {
          ArgVector.push_back(SaveStringInSet(SavedStrings, CurArg));
        } else {
          ExpandArgsFromBuf(CurArg.c_str(), ArgVector, SavedStrings);
        }

        CurArg = "";
      }
      if (*P == '\0')
        break;
      else
        continue;
    }

    if (isspace(*P)) {
      if (InQuote != ' ')
        CurArg.push_back(*P);
      continue;
    }

    if (*P == '"' || *P == '\'') {
      if (InQuote == *P)
        InQuote = ' ';
      else if (InQuote == ' ')
        InQuote = *P;
      else
        CurArg.push_back(*P);
      continue;
    }

    if (*P == '\\') {
      ++P;
      if (*P != '\0')
        CurArg.push_back(*P);
      continue;
    }
    CurArg.push_back(*P);
  }
}

static void ExpandArgv(int argc, const char **argv,
                       SmallVectorImpl<const char*> &ArgVector,
                       std::set<std::string> &SavedStrings) {
  for (int i = 0; i < argc; ++i) {
    const char *Arg = argv[i];
    if (Arg[0] != '@') {
      ArgVector.push_back(SaveStringInSet(SavedStrings, std::string(Arg)));
      continue;
    }

    ExpandArgsFromBuf(Arg, ArgVector, SavedStrings);
  }
}

static void ParseProgName(SmallVectorImpl<const char *> &ArgVector,
                          std::set<std::string> &SavedStrings,
                          Driver &TheDriver)
{
  // Try to infer frontend type and default target from the program name.

  // suffixes[] contains the list of known driver suffixes.
  // Suffixes are compared against the program name in order.
  // If there is a match, the frontend type is updated as necessary (CPP/C++).
  // If there is no match, a second round is done after stripping the last
  // hyphen and everything following it. This allows using something like
  // "clang++-2.9".

  // If there is a match in either the first or second round,
  // the function tries to identify a target as prefix. E.g.
  // "x86_64-linux-clang" as interpreted as suffix "clang" with
  // target prefix "x86_64-linux". If such a target prefix is found,
  // is gets added via -target as implicit first argument.
  static const struct {
    const char *Suffix;
    bool IsCXX;
    bool IsCPP;
  } suffixes [] = {
    { "clang", false, false },
    { "clang++", true, false },
    { "clang-c++", true, false },
    { "clang-cc", false, false },
    { "clang-cpp", false, true },
    { "clang-g++", true, false },
    { "clang-gcc", false, false },
    { "cc", false, false },
    { "cpp", false, true },
    { "++", true, false },
  };
  std::string ProgName(llvm::sys::path::stem(ArgVector[0]));
  StringRef ProgNameRef(ProgName);
  StringRef Prefix;

  for (int Components = 2; Components; --Components) {
    bool FoundMatch = false;
    size_t i;

    for (i = 0; i < sizeof(suffixes) / sizeof(suffixes[0]); ++i) {
      if (ProgNameRef.endswith(suffixes[i].Suffix)) {
        FoundMatch = true;
        if (suffixes[i].IsCXX)
          TheDriver.CCCIsCXX = true;
        if (suffixes[i].IsCPP)
          TheDriver.CCCIsCPP = true;
        break;
      }
    }

    if (FoundMatch) {
      StringRef::size_type LastComponent = ProgNameRef.rfind('-',
        ProgNameRef.size() - strlen(suffixes[i].Suffix));
      if (LastComponent != StringRef::npos)
        Prefix = ProgNameRef.slice(0, LastComponent);
      break;
    }

    StringRef::size_type LastComponent = ProgNameRef.rfind('-');
    if (LastComponent == StringRef::npos)
      break;
    ProgNameRef = ProgNameRef.slice(0, LastComponent);
  }

  if (Prefix.empty())
    return;

  std::string IgnoredError;
  if (llvm::TargetRegistry::lookupTarget(Prefix, IgnoredError)) {
    SmallVectorImpl<const char *>::iterator it = ArgVector.begin();
    if (it != ArgVector.end())
      ++it;
    ArgVector.insert(it, SaveStringInSet(SavedStrings, Prefix));
    ArgVector.insert(it,
      SaveStringInSet(SavedStrings, std::string("-target")));
  }
}

int main(int argc_, const char **argv_) {
  llvm::sys::PrintStackTraceOnErrorSignal();
  llvm::PrettyStackTraceProgram X(argc_, argv_);

  std::set<std::string> SavedStrings;
  SmallVector<const char*, 256> argv;

  ExpandArgv(argc_, argv_, argv, SavedStrings);

  // Handle -cc1 integrated tools.
  if (argv.size() > 1 && StringRef(argv[1]).startswith("-cc1")) {
    StringRef Tool = argv[1] + 4;

    if (Tool == "")
      return cc1_main(argv.data()+2, argv.data()+argv.size(), argv[0],
                      (void*) (intptr_t) GetExecutablePath);
    if (Tool == "as")
      return cc1as_main(argv.data()+2, argv.data()+argv.size(), argv[0],
                      (void*) (intptr_t) GetExecutablePath);

    // Reject unknown tools.
    llvm::errs() << "error: unknown integrated tool '" << Tool << "'\n";
    return 1;
  }

  bool CanonicalPrefixes = true;
  for (int i = 1, size = argv.size(); i < size; ++i) {
    if (StringRef(argv[i]) == "-no-canonical-prefixes") {
      CanonicalPrefixes = false;
      break;
    }
  }

  llvm::sys::Path Path = GetExecutablePath(argv[0], CanonicalPrefixes);

  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts = new DiagnosticOptions;
  {
    // Note that ParseDiagnosticArgs() uses the cc1 option table.
    OwningPtr<OptTable> CC1Opts(createDriverOptTable());
    unsigned MissingArgIndex, MissingArgCount;
    OwningPtr<InputArgList> Args(CC1Opts->ParseArgs(argv.begin()+1, argv.end(),
                                            MissingArgIndex, MissingArgCount));
    // We ignore MissingArgCount and the return value of ParseDiagnosticArgs.
    // Any errors that would be diagnosed here will also be diagnosed later,
    // when the DiagnosticsEngine actually exists.
    (void) ParseDiagnosticArgs(*DiagOpts, *Args);
  }
  // Now we can create the DiagnosticsEngine with a properly-filled-out
  // DiagnosticOptions instance.
  TextDiagnosticPrinter *DiagClient
    = new TextDiagnosticPrinter(llvm::errs(), &*DiagOpts);
  DiagClient->setPrefix(llvm::sys::path::stem(Path.str()));
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID(new DiagnosticIDs());

  DiagnosticsEngine Diags(DiagID, &*DiagOpts, DiagClient);
  ProcessWarningOptions(Diags, *DiagOpts);

#ifdef CLANG_IS_PRODUCTION
  const bool IsProduction = true;
#else
  const bool IsProduction = false;
#endif
  Driver TheDriver(Path.str(), llvm::sys::getDefaultTargetTriple(),
                   "a.out", IsProduction, Diags);

  // Attempt to find the original path used to invoke the driver, to determine
  // the installed path. We do this manually, because we want to support that
  // path being a symlink.
  {
    SmallString<128> InstalledPath(argv[0]);

    // Do a PATH lookup, if there are no directory components.
    if (llvm::sys::path::filename(InstalledPath) == InstalledPath) {
      llvm::sys::Path Tmp = llvm::sys::Program::FindProgramByName(
        llvm::sys::path::filename(InstalledPath.str()));
      if (!Tmp.empty())
        InstalledPath = Tmp.str();
    }
    llvm::sys::fs::make_absolute(InstalledPath);
    InstalledPath = llvm::sys::path::parent_path(InstalledPath);
    bool exists;
    if (!llvm::sys::fs::exists(InstalledPath.str(), exists) && exists)
      TheDriver.setInstalledDir(InstalledPath);
  }

  llvm::InitializeAllTargets();
  ParseProgName(argv, SavedStrings, TheDriver);

  // Handle CC_PRINT_OPTIONS and CC_PRINT_OPTIONS_FILE.
  TheDriver.CCPrintOptions = !!::getenv("CC_PRINT_OPTIONS");
  if (TheDriver.CCPrintOptions)
    TheDriver.CCPrintOptionsFilename = ::getenv("CC_PRINT_OPTIONS_FILE");

  // Handle CC_PRINT_HEADERS and CC_PRINT_HEADERS_FILE.
  TheDriver.CCPrintHeaders = !!::getenv("CC_PRINT_HEADERS");
  if (TheDriver.CCPrintHeaders)
    TheDriver.CCPrintHeadersFilename = ::getenv("CC_PRINT_HEADERS_FILE");

  // Handle CC_LOG_DIAGNOSTICS and CC_LOG_DIAGNOSTICS_FILE.
  TheDriver.CCLogDiagnostics = !!::getenv("CC_LOG_DIAGNOSTICS");
  if (TheDriver.CCLogDiagnostics)
    TheDriver.CCLogDiagnosticsFilename = ::getenv("CC_LOG_DIAGNOSTICS_FILE");

  // Handle QA_OVERRIDE_GCC3_OPTIONS and CCC_ADD_ARGS, used for editing a
  // command line behind the scenes.
  if (const char *OverrideStr = ::getenv("QA_OVERRIDE_GCC3_OPTIONS")) {
    // FIXME: Driver shouldn't take extra initial argument.
    ApplyQAOverride(argv, OverrideStr, SavedStrings);
  } else if (const char *Cur = ::getenv("CCC_ADD_ARGS")) {
    // FIXME: Driver shouldn't take extra initial argument.
    std::vector<const char*> ExtraArgs;

    for (;;) {
      const char *Next = strchr(Cur, ',');

      if (Next) {
        ExtraArgs.push_back(SaveStringInSet(SavedStrings,
                                            std::string(Cur, Next)));
        Cur = Next + 1;
      } else {
        if (*Cur != '\0')
          ExtraArgs.push_back(SaveStringInSet(SavedStrings, Cur));
        break;
      }
    }

    argv.insert(&argv[1], ExtraArgs.begin(), ExtraArgs.end());
  }

  OwningPtr<Compilation> C(TheDriver.BuildCompilation(argv));
  int Res = 0;
  const Command *FailingCommand = 0;
  if (C.get())
    Res = TheDriver.ExecuteCompilation(*C, FailingCommand);

  // Force a crash to test the diagnostics.
  if(::getenv("FORCE_CLANG_DIAGNOSTICS_CRASH"))
     Res = -1;

  // If result status is < 0, then the driver command signalled an error.
  // In this case, generate additional diagnostic information if possible.
  if (Res < 0)
    TheDriver.generateCompilationDiagnostics(*C, FailingCommand);

  // If any timers were active but haven't been destroyed yet, print their
  // results now.  This happens in -disable-free mode.
  llvm::TimerGroup::printAll(llvm::errs());
  
  llvm::llvm_shutdown();

#ifdef _WIN32
  // Exit status should not be negative on Win32, unless abnormal termination.
  // Once abnormal termiation was caught, negative status should not be
  // propagated.
  if (Res < 0)
    Res = 1;
#endif

  return Res;
}
