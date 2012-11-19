//===--- Driver.cpp - Clang GCC Compatible Driver -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Driver.h"

#include "clang/Driver/Action.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Job.h"
#include "clang/Driver/OptTable.h"
#include "clang/Driver/Option.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/Tool.h"
#include "clang/Driver/ToolChain.h"

#include "clang/Basic/Version.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"

#include "InputInfo.h"
#include "ToolChains.h"

#include <map>

#include "clang/Config/config.h"

using namespace clang::driver;
using namespace clang;

Driver::Driver(StringRef ClangExecutable,
               StringRef DefaultTargetTriple,
               StringRef DefaultImageName,
               bool IsProduction,
               DiagnosticsEngine &Diags)
  : Opts(createDriverOptTable()), Diags(Diags),
    ClangExecutable(ClangExecutable), SysRoot(DEFAULT_SYSROOT),
    UseStdLib(true), DefaultTargetTriple(DefaultTargetTriple),
    DefaultImageName(DefaultImageName),
    DriverTitle("clang LLVM compiler"),
    CCPrintOptionsFilename(0), CCPrintHeadersFilename(0),
    CCLogDiagnosticsFilename(0), CCCIsCXX(false),
    CCCIsCPP(false),CCCEcho(false), CCCPrintBindings(false),
    CCPrintOptions(false), CCPrintHeaders(false), CCLogDiagnostics(false),
    CCGenDiagnostics(false), CCCGenericGCCName(""), CheckInputsExist(true),
    CCCUsePCH(true), SuppressMissingInputWarning(false) {

  Name = llvm::sys::path::stem(ClangExecutable);
  Dir  = llvm::sys::path::parent_path(ClangExecutable);

  // Compute the path to the resource directory.
  StringRef ClangResourceDir(CLANG_RESOURCE_DIR);
  SmallString<128> P(Dir);
  if (ClangResourceDir != "")
    llvm::sys::path::append(P, ClangResourceDir);
  else
    llvm::sys::path::append(P, "..", "lib", "clang", CLANG_VERSION_STRING);
  ResourceDir = P.str();
}

Driver::~Driver() {
  delete Opts;

  for (llvm::StringMap<ToolChain *>::iterator I = ToolChains.begin(),
                                              E = ToolChains.end();
       I != E; ++I)
    delete I->second;
}

InputArgList *Driver::ParseArgStrings(ArrayRef<const char *> ArgList) {
  llvm::PrettyStackTraceString CrashInfo("Command line argument parsing");
  unsigned MissingArgIndex, MissingArgCount;
  InputArgList *Args = getOpts().ParseArgs(ArgList.begin(), ArgList.end(),
                                           MissingArgIndex, MissingArgCount);

  // Check for missing argument error.
  if (MissingArgCount)
    Diag(clang::diag::err_drv_missing_argument)
      << Args->getArgString(MissingArgIndex) << MissingArgCount;

  // Check for unsupported options.
  for (ArgList::const_iterator it = Args->begin(), ie = Args->end();
       it != ie; ++it) {
    Arg *A = *it;
    if (A->getOption().hasFlag(options::Unsupported)) {
      Diag(clang::diag::err_drv_unsupported_opt) << A->getAsString(*Args);
      continue;
    }

    // Warn about -mcpu= without an argument.
    if (A->getOption().matches(options::OPT_mcpu_EQ) &&
        A->containsValue("")) {
      Diag(clang::diag::warn_drv_empty_joined_argument) <<
        A->getAsString(*Args);
    }
  }

  return Args;
}

// Determine which compilation mode we are in. We look for options which
// affect the phase, starting with the earliest phases, and record which
// option we used to determine the final phase.
phases::ID Driver::getFinalPhase(const DerivedArgList &DAL, Arg **FinalPhaseArg)
const {
  Arg *PhaseArg = 0;
  phases::ID FinalPhase;

  // -{E,M,MM} only run the preprocessor.
  if (CCCIsCPP ||
      (PhaseArg = DAL.getLastArg(options::OPT_E)) ||
      (PhaseArg = DAL.getLastArg(options::OPT_M, options::OPT_MM))) {
    FinalPhase = phases::Preprocess;

    // -{fsyntax-only,-analyze,emit-ast,S} only run up to the compiler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_fsyntax_only)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_rewrite_legacy_objc)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__migrate)) ||
             (PhaseArg = DAL.getLastArg(options::OPT__analyze,
                                        options::OPT__analyze_auto)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_emit_ast)) ||
             (PhaseArg = DAL.getLastArg(options::OPT_S))) {
    FinalPhase = phases::Compile;

    // -c only runs up to the assembler.
  } else if ((PhaseArg = DAL.getLastArg(options::OPT_c))) {
    FinalPhase = phases::Assemble;

    // Otherwise do everything.
  } else
    FinalPhase = phases::Link;

  if (FinalPhaseArg)
    *FinalPhaseArg = PhaseArg;

  return FinalPhase;
}

DerivedArgList *Driver::TranslateInputArgs(const InputArgList &Args) const {
  DerivedArgList *DAL = new DerivedArgList(Args);

  bool HasNostdlib = Args.hasArg(options::OPT_nostdlib);
  for (ArgList::const_iterator it = Args.begin(),
         ie = Args.end(); it != ie; ++it) {
    const Arg *A = *it;

    // Unfortunately, we have to parse some forwarding options (-Xassembler,
    // -Xlinker, -Xpreprocessor) because we either integrate their functionality
    // (assembler and preprocessor), or bypass a previous driver ('collect2').

    // Rewrite linker options, to replace --no-demangle with a custom internal
    // option.
    if ((A->getOption().matches(options::OPT_Wl_COMMA) ||
         A->getOption().matches(options::OPT_Xlinker)) &&
        A->containsValue("--no-demangle")) {
      // Add the rewritten no-demangle argument.
      DAL->AddFlagArg(A, Opts->getOption(options::OPT_Z_Xlinker__no_demangle));

      // Add the remaining values as Xlinker arguments.
      for (unsigned i = 0, e = A->getNumValues(); i != e; ++i)
        if (StringRef(A->getValue(i)) != "--no-demangle")
          DAL->AddSeparateArg(A, Opts->getOption(options::OPT_Xlinker),
                              A->getValue(i));

      continue;
    }

    // Rewrite preprocessor options, to replace -Wp,-MD,FOO which is used by
    // some build systems. We don't try to be complete here because we don't
    // care to encourage this usage model.
    if (A->getOption().matches(options::OPT_Wp_COMMA) &&
        (A->getValue(0) == StringRef("-MD") ||
         A->getValue(0) == StringRef("-MMD"))) {
      // Rewrite to -MD/-MMD along with -MF.
      if (A->getValue(0) == StringRef("-MD"))
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_MD));
      else
        DAL->AddFlagArg(A, Opts->getOption(options::OPT_MMD));
      if (A->getNumValues() == 2)
        DAL->AddSeparateArg(A, Opts->getOption(options::OPT_MF),
                            A->getValue(1));
      continue;
    }

    // Rewrite reserved library names.
    if (A->getOption().matches(options::OPT_l)) {
      StringRef Value = A->getValue();

      // Rewrite unless -nostdlib is present.
      if (!HasNostdlib && Value == "stdc++") {
        DAL->AddFlagArg(A, Opts->getOption(
                              options::OPT_Z_reserved_lib_stdcxx));
        continue;
      }

      // Rewrite unconditionally.
      if (Value == "cc_kext") {
        DAL->AddFlagArg(A, Opts->getOption(
                              options::OPT_Z_reserved_lib_cckext));
        continue;
      }
    }

    DAL->append(*it);
  }

  // Add a default value of -mlinker-version=, if one was given and the user
  // didn't specify one.
#if defined(HOST_LINK_VERSION)
  if (!Args.hasArg(options::OPT_mlinker_version_EQ)) {
    DAL->AddJoinedArg(0, Opts->getOption(options::OPT_mlinker_version_EQ),
                      HOST_LINK_VERSION);
    DAL->getLastArg(options::OPT_mlinker_version_EQ)->claim();
  }
#endif

  return DAL;
}

Compilation *Driver::BuildCompilation(ArrayRef<const char *> ArgList) {
  llvm::PrettyStackTraceString CrashInfo("Compilation construction");

  // FIXME: Handle environment options which affect driver behavior, somewhere
  // (client?). GCC_EXEC_PREFIX, LPATH, CC_PRINT_OPTIONS.

  if (char *env = ::getenv("COMPILER_PATH")) {
    StringRef CompilerPath = env;
    while (!CompilerPath.empty()) {
      std::pair<StringRef, StringRef> Split = CompilerPath.split(':');
      PrefixDirs.push_back(Split.first);
      CompilerPath = Split.second;
    }
  }

  // FIXME: What are we going to do with -V and -b?

  // FIXME: This stuff needs to go into the Compilation, not the driver.
  bool CCCPrintOptions = false, CCCPrintActions = false;

  InputArgList *Args = ParseArgStrings(ArgList.slice(1));

  // -no-canonical-prefixes is used very early in main.
  Args->ClaimAllArgs(options::OPT_no_canonical_prefixes);

  // Ignore -pipe.
  Args->ClaimAllArgs(options::OPT_pipe);

  // Extract -ccc args.
  //
  // FIXME: We need to figure out where this behavior should live. Most of it
  // should be outside in the client; the parts that aren't should have proper
  // options, either by introducing new ones or by overloading gcc ones like -V
  // or -b.
  CCCPrintOptions = Args->hasArg(options::OPT_ccc_print_options);
  CCCPrintActions = Args->hasArg(options::OPT_ccc_print_phases);
  CCCPrintBindings = Args->hasArg(options::OPT_ccc_print_bindings);
  CCCIsCXX = Args->hasArg(options::OPT_ccc_cxx) || CCCIsCXX;
  CCCEcho = Args->hasArg(options::OPT_ccc_echo);
  if (const Arg *A = Args->getLastArg(options::OPT_ccc_gcc_name))
    CCCGenericGCCName = A->getValue();
  CCCUsePCH = Args->hasFlag(options::OPT_ccc_pch_is_pch,
                            options::OPT_ccc_pch_is_pth);
  // FIXME: DefaultTargetTriple is used by the target-prefixed calls to as/ld
  // and getToolChain is const.
  if (const Arg *A = Args->getLastArg(options::OPT_target))
    DefaultTargetTriple = A->getValue();
  if (const Arg *A = Args->getLastArg(options::OPT_ccc_install_dir))
    Dir = InstalledDir = A->getValue();
  for (arg_iterator it = Args->filtered_begin(options::OPT_B),
         ie = Args->filtered_end(); it != ie; ++it) {
    const Arg *A = *it;
    A->claim();
    PrefixDirs.push_back(A->getValue(0));
  }
  if (const Arg *A = Args->getLastArg(options::OPT__sysroot_EQ))
    SysRoot = A->getValue();
  if (Args->hasArg(options::OPT_nostdlib))
    UseStdLib = false;

  // Perform the default argument translations.
  DerivedArgList *TranslatedArgs = TranslateInputArgs(*Args);

  // Owned by the host.
  const ToolChain &TC = getToolChain(*Args);

  // The compilation takes ownership of Args.
  Compilation *C = new Compilation(*this, TC, Args, TranslatedArgs);

  // FIXME: This behavior shouldn't be here.
  if (CCCPrintOptions) {
    PrintOptions(C->getInputArgs());
    return C;
  }

  if (!HandleImmediateArgs(*C))
    return C;

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C->getDefaultToolChain(), C->getArgs(), Inputs);

  // Construct the list of abstract actions to perform for this compilation. On
  // Darwin target OSes this uses the driver-driver and universal actions.
  if (TC.getTriple().isOSDarwin())
    BuildUniversalActions(C->getDefaultToolChain(), C->getArgs(),
                          Inputs, C->getActions());
  else
    BuildActions(C->getDefaultToolChain(), C->getArgs(), Inputs,
                 C->getActions());

  if (CCCPrintActions) {
    PrintActions(*C);
    return C;
  }

  BuildJobs(*C);

  return C;
}

// When clang crashes, produce diagnostic information including the fully
// preprocessed source file(s).  Request that the developer attach the
// diagnostic information to a bug report.
void Driver::generateCompilationDiagnostics(Compilation &C,
                                            const Command *FailingCommand) {
  if (C.getArgs().hasArg(options::OPT_fno_crash_diagnostics))
    return;

  // Don't try to generate diagnostics for link jobs.
  if (FailingCommand && FailingCommand->getCreator().isLinkJob())
    return;

  // Print the version of the compiler.
  PrintVersion(C, llvm::errs());

  Diag(clang::diag::note_drv_command_failed_diag_msg)
    << "PLEASE submit a bug report to " BUG_REPORT_URL " and include the "
    "crash backtrace, preprocessed source, and associated run script.";

  // Suppress driver output and emit preprocessor output to temp file.
  CCCIsCPP = true;
  CCGenDiagnostics = true;
  C.getArgs().AddFlagArg(0, Opts->getOption(options::OPT_frewrite_includes));

  // Save the original job command(s).
  std::string Cmd;
  llvm::raw_string_ostream OS(Cmd);
  if (FailingCommand)
    C.PrintDiagnosticJob(OS, *FailingCommand);
  else
    // Crash triggered by FORCE_CLANG_DIAGNOSTICS_CRASH, which doesn't have an
    // associated FailingCommand, so just pass all jobs.
    C.PrintDiagnosticJob(OS, C.getJobs());
  OS.flush();

  // Clear stale state and suppress tool output.
  C.initCompilationForDiagnostics();
  Diags.Reset();

  // Construct the list of inputs.
  InputList Inputs;
  BuildInputs(C.getDefaultToolChain(), C.getArgs(), Inputs);

  for (InputList::iterator it = Inputs.begin(), ie = Inputs.end(); it != ie;) {
    bool IgnoreInput = false;

    // Ignore input from stdin or any inputs that cannot be preprocessed.
    if (!strcmp(it->second->getValue(), "-")) {
      Diag(clang::diag::note_drv_command_failed_diag_msg)
        << "Error generating preprocessed source(s) - ignoring input from stdin"
        ".";
      IgnoreInput = true;
    } else if (types::getPreprocessedType(it->first) == types::TY_INVALID) {
      IgnoreInput = true;
    }

    if (IgnoreInput) {
      it = Inputs.erase(it);
      ie = Inputs.end();
    } else {
      ++it;
    }
  }

  // Don't attempt to generate preprocessed files if multiple -arch options are
  // used, unless they're all duplicates.
  llvm::StringSet<> ArchNames;
  for (ArgList::const_iterator it = C.getArgs().begin(), ie = C.getArgs().end();
       it != ie; ++it) {
    Arg *A = *it;
    if (A->getOption().matches(options::OPT_arch)) {
      StringRef ArchName = A->getValue();
      ArchNames.insert(ArchName);
    }
  }
  if (ArchNames.size() > 1) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "Error generating preprocessed source(s) - cannot generate "
      "preprocessed source with multiple -arch options.";
    return;
  }

  if (Inputs.empty()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "Error generating preprocessed source(s) - no preprocessable inputs.";
    return;
  }

  // Construct the list of abstract actions to perform for this compilation. On
  // Darwin OSes this uses the driver-driver and builds universal actions.
  const ToolChain &TC = C.getDefaultToolChain();
  if (TC.getTriple().isOSDarwin())
    BuildUniversalActions(TC, C.getArgs(), Inputs, C.getActions());
  else
    BuildActions(TC, C.getArgs(), Inputs, C.getActions());

  BuildJobs(C);

  // If there were errors building the compilation, quit now.
  if (Diags.hasErrorOccurred()) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "Error generating preprocessed source(s).";
    return;
  }

  // Generate preprocessed output.
  FailingCommand = 0;
  int Res = C.ExecuteJob(C.getJobs(), FailingCommand);

  // If the command succeeded, we are done.
  if (Res == 0) {
    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "\n********************\n\n"
      "PLEASE ATTACH THE FOLLOWING FILES TO THE BUG REPORT:\n"
      "Preprocessed source(s) and associated run script(s) are located at:";
    ArgStringList Files = C.getTempFiles();
    for (ArgStringList::const_iterator it = Files.begin(), ie = Files.end();
         it != ie; ++it) {
      Diag(clang::diag::note_drv_command_failed_diag_msg) << *it;

      std::string Err;
      std::string Script = StringRef(*it).rsplit('.').first;
      Script += ".sh";
      llvm::raw_fd_ostream ScriptOS(Script.c_str(), Err,
                                    llvm::raw_fd_ostream::F_Excl |
                                    llvm::raw_fd_ostream::F_Binary);
      if (!Err.empty()) {
        Diag(clang::diag::note_drv_command_failed_diag_msg)
          << "Error generating run script: " + Script + " " + Err;
      } else {
        // Append the new filename with correct preprocessed suffix.
        size_t I, E;
        I = Cmd.find("-main-file-name ");
        assert (I != std::string::npos && "Expected to find -main-file-name");
        I += 16;
        E = Cmd.find(" ", I);
        assert (E != std::string::npos && "-main-file-name missing argument?");
        StringRef OldFilename = StringRef(Cmd).slice(I, E);
        StringRef NewFilename = llvm::sys::path::filename(*it);
        I = StringRef(Cmd).rfind(OldFilename);
        E = I + OldFilename.size();
        I = Cmd.rfind(" ", I) + 1;
        Cmd.replace(I, E - I, NewFilename.data(), NewFilename.size());
        ScriptOS << Cmd;
        Diag(clang::diag::note_drv_command_failed_diag_msg) << Script;
      }
    }
    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "\n\n********************";
  } else {
    // Failure, remove preprocessed files.
    if (!C.getArgs().hasArg(options::OPT_save_temps))
      C.CleanupFileList(C.getTempFiles(), true);

    Diag(clang::diag::note_drv_command_failed_diag_msg)
      << "Error generating preprocessed source(s).";
  }
}

int Driver::ExecuteCompilation(const Compilation &C,
                               const Command *&FailingCommand) const {
  // Just print if -### was present.
  if (C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    C.PrintJob(llvm::errs(), C.getJobs(), "\n", true);
    return 0;
  }

  // If there were errors building the compilation, quit now.
  if (Diags.hasErrorOccurred())
    return 1;

  int Res = C.ExecuteJob(C.getJobs(), FailingCommand);

  // Remove temp files.
  C.CleanupFileList(C.getTempFiles());

  // If the command succeeded, we are done.
  if (Res == 0)
    return Res;

  // Otherwise, remove result files as well.
  if (!C.getArgs().hasArg(options::OPT_save_temps)) {
    C.CleanupFileList(C.getResultFiles(), true);

    // Failure result files are valid unless we crashed.
    if (Res < 0)
      C.CleanupFileList(C.getFailureResultFiles(), true);
  }

  // Print extra information about abnormal failures, if possible.
  //
  // This is ad-hoc, but we don't want to be excessively noisy. If the result
  // status was 1, assume the command failed normally. In particular, if it was
  // the compiler then assume it gave a reasonable error code. Failures in other
  // tools are less common, and they generally have worse diagnostics, so always
  // print the diagnostic there.
  const Tool &FailingTool = FailingCommand->getCreator();

  if (!FailingCommand->getCreator().hasGoodDiagnostics() || Res != 1) {
    // FIXME: See FIXME above regarding result code interpretation.
    if (Res < 0)
      Diag(clang::diag::err_drv_command_signalled)
        << FailingTool.getShortName();
    else
      Diag(clang::diag::err_drv_command_failed)
        << FailingTool.getShortName() << Res;
  }

  return Res;
}

void Driver::PrintOptions(const ArgList &Args) const {
  unsigned i = 0;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end();
       it != ie; ++it, ++i) {
    Arg *A = *it;
    llvm::errs() << "Option " << i << " - "
                 << "Name: \"" << A->getOption().getPrefixedName() << "\", "
                 << "Values: {";
    for (unsigned j = 0; j < A->getNumValues(); ++j) {
      if (j)
        llvm::errs() << ", ";
      llvm::errs() << '"' << A->getValue(j) << '"';
    }
    llvm::errs() << "}\n";
  }
}

void Driver::PrintHelp(bool ShowHidden) const {
  getOpts().PrintHelp(llvm::outs(), Name.c_str(), DriverTitle.c_str(),
                      /*Include*/0,
                      /*Exclude*/options::NoDriverOption |
                      (ShowHidden ? 0 : options::HelpHidden));
}

void Driver::PrintVersion(const Compilation &C, raw_ostream &OS) const {
  // FIXME: The following handlers should use a callback mechanism, we don't
  // know what the client would like to do.
  OS << getClangFullVersion() << '\n';
  const ToolChain &TC = C.getDefaultToolChain();
  OS << "Target: " << TC.getTripleString() << '\n';

  // Print the threading model.
  //
  // FIXME: Implement correctly.
  OS << "Thread model: " << "posix" << '\n';
}

/// PrintDiagnosticCategories - Implement the --print-diagnostic-categories
/// option.
static void PrintDiagnosticCategories(raw_ostream &OS) {
  // Skip the empty category.
  for (unsigned i = 1, max = DiagnosticIDs::getNumberOfCategories();
       i != max; ++i)
    OS << i << ',' << DiagnosticIDs::getCategoryNameFromID(i) << '\n';
}

bool Driver::HandleImmediateArgs(const Compilation &C) {
  // The order these options are handled in gcc is all over the place, but we
  // don't expect inconsistencies w.r.t. that to matter in practice.

  if (C.getArgs().hasArg(options::OPT_dumpmachine)) {
    llvm::outs() << C.getDefaultToolChain().getTripleString() << '\n';
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_dumpversion)) {
    // Since -dumpversion is only implemented for pedantic GCC compatibility, we
    // return an answer which matches our definition of __VERSION__.
    //
    // If we want to return a more correct answer some day, then we should
    // introduce a non-pedantically GCC compatible mode to Clang in which we
    // provide sensible definitions for -dumpversion, __VERSION__, etc.
    llvm::outs() << "4.2.1\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__print_diagnostic_categories)) {
    PrintDiagnosticCategories(llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_help) ||
      C.getArgs().hasArg(options::OPT__help_hidden)) {
    PrintHelp(C.getArgs().hasArg(options::OPT__help_hidden));
    return false;
  }

  if (C.getArgs().hasArg(options::OPT__version)) {
    // Follow gcc behavior and use stdout for --version and stderr for -v.
    PrintVersion(C, llvm::outs());
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_v) ||
      C.getArgs().hasArg(options::OPT__HASH_HASH_HASH)) {
    PrintVersion(C, llvm::errs());
    SuppressMissingInputWarning = true;
  }

  const ToolChain &TC = C.getDefaultToolChain();
  if (C.getArgs().hasArg(options::OPT_print_search_dirs)) {
    llvm::outs() << "programs: =";
    for (ToolChain::path_list::const_iterator it = TC.getProgramPaths().begin(),
           ie = TC.getProgramPaths().end(); it != ie; ++it) {
      if (it != TC.getProgramPaths().begin())
        llvm::outs() << ':';
      llvm::outs() << *it;
    }
    llvm::outs() << "\n";
    llvm::outs() << "libraries: =" << ResourceDir;

    StringRef sysroot = C.getSysRoot();

    for (ToolChain::path_list::const_iterator it = TC.getFilePaths().begin(),
           ie = TC.getFilePaths().end(); it != ie; ++it) {
      llvm::outs() << ':';
      const char *path = it->c_str();
      if (path[0] == '=')
        llvm::outs() << sysroot << path + 1;
      else
        llvm::outs() << path;
    }
    llvm::outs() << "\n";
    return false;
  }

  // FIXME: The following handlers should use a callback mechanism, we don't
  // know what the client would like to do.
  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_file_name_EQ)) {
    llvm::outs() << GetFilePath(A->getValue(), TC) << "\n";
    return false;
  }

  if (Arg *A = C.getArgs().getLastArg(options::OPT_print_prog_name_EQ)) {
    llvm::outs() << GetProgramPath(A->getValue(), TC) << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_libgcc_file_name)) {
    llvm::outs() << GetFilePath("libgcc.a", TC) << "\n";
    return false;
  }

  if (C.getArgs().hasArg(options::OPT_print_multi_lib)) {
    // FIXME: We need tool chain support for this.
    llvm::outs() << ".;\n";

    switch (C.getDefaultToolChain().getTriple().getArch()) {
    default:
      break;

    case llvm::Triple::x86_64:
      llvm::outs() << "x86_64;@m64" << "\n";
      break;

    case llvm::Triple::ppc64:
      llvm::outs() << "ppc64;@m64" << "\n";
      break;
    }
    return false;
  }

  // FIXME: What is the difference between print-multi-directory and
  // print-multi-os-directory?
  if (C.getArgs().hasArg(options::OPT_print_multi_directory) ||
      C.getArgs().hasArg(options::OPT_print_multi_os_directory)) {
    switch (C.getDefaultToolChain().getTriple().getArch()) {
    default:
    case llvm::Triple::x86:
    case llvm::Triple::ppc:
      llvm::outs() << "." << "\n";
      break;

    case llvm::Triple::x86_64:
      llvm::outs() << "x86_64" << "\n";
      break;

    case llvm::Triple::ppc64:
      llvm::outs() << "ppc64" << "\n";
      break;
    }
    return false;
  }

  return true;
}

static unsigned PrintActions1(const Compilation &C, Action *A,
                              std::map<Action*, unsigned> &Ids) {
  if (Ids.count(A))
    return Ids[A];

  std::string str;
  llvm::raw_string_ostream os(str);

  os << Action::getClassName(A->getKind()) << ", ";
  if (InputAction *IA = dyn_cast<InputAction>(A)) {
    os << "\"" << IA->getInputArg().getValue() << "\"";
  } else if (BindArchAction *BIA = dyn_cast<BindArchAction>(A)) {
    os << '"' << BIA->getArchName() << '"'
       << ", {" << PrintActions1(C, *BIA->begin(), Ids) << "}";
  } else {
    os << "{";
    for (Action::iterator it = A->begin(), ie = A->end(); it != ie;) {
      os << PrintActions1(C, *it, Ids);
      ++it;
      if (it != ie)
        os << ", ";
    }
    os << "}";
  }

  unsigned Id = Ids.size();
  Ids[A] = Id;
  llvm::errs() << Id << ": " << os.str() << ", "
               << types::getTypeName(A->getType()) << "\n";

  return Id;
}

void Driver::PrintActions(const Compilation &C) const {
  std::map<Action*, unsigned> Ids;
  for (ActionList::const_iterator it = C.getActions().begin(),
         ie = C.getActions().end(); it != ie; ++it)
    PrintActions1(C, *it, Ids);
}

/// \brief Check whether the given input tree contains any compilation or
/// assembly actions.
static bool ContainsCompileOrAssembleAction(const Action *A) {
  if (isa<CompileJobAction>(A) || isa<AssembleJobAction>(A))
    return true;

  for (Action::const_iterator it = A->begin(), ie = A->end(); it != ie; ++it)
    if (ContainsCompileOrAssembleAction(*it))
      return true;

  return false;
}

void Driver::BuildUniversalActions(const ToolChain &TC,
                                   const DerivedArgList &Args,
                                   const InputList &BAInputs,
                                   ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building universal build actions");
  // Collect the list of architectures. Duplicates are allowed, but should only
  // be handled once (in the order seen).
  llvm::StringSet<> ArchNames;
  SmallVector<const char *, 4> Archs;
  for (ArgList::const_iterator it = Args.begin(), ie = Args.end();
       it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().matches(options::OPT_arch)) {
      // Validate the option here; we don't save the type here because its
      // particular spelling may participate in other driver choices.
      llvm::Triple::ArchType Arch =
        tools::darwin::getArchTypeForDarwinArchName(A->getValue());
      if (Arch == llvm::Triple::UnknownArch) {
        Diag(clang::diag::err_drv_invalid_arch_name)
          << A->getAsString(Args);
        continue;
      }

      A->claim();
      if (ArchNames.insert(A->getValue()))
        Archs.push_back(A->getValue());
    }
  }

  // When there is no explicit arch for this platform, make sure we still bind
  // the architecture (to the default) so that -Xarch_ is handled correctly.
  if (!Archs.size())
    Archs.push_back(Args.MakeArgString(TC.getDefaultUniversalArchName()));

  // FIXME: We killed off some others but these aren't yet detected in a
  // functional manner. If we added information to jobs about which "auxiliary"
  // files they wrote then we could detect the conflict these cause downstream.
  if (Archs.size() > 1) {
    // No recovery needed, the point of this is just to prevent
    // overwriting the same files.
    if (const Arg *A = Args.getLastArg(options::OPT_save_temps))
      Diag(clang::diag::err_drv_invalid_opt_with_multiple_archs)
        << A->getAsString(Args);
  }

  ActionList SingleActions;
  BuildActions(TC, Args, BAInputs, SingleActions);

  // Add in arch bindings for every top level action, as well as lipo and
  // dsymutil steps if needed.
  for (unsigned i = 0, e = SingleActions.size(); i != e; ++i) {
    Action *Act = SingleActions[i];

    // Make sure we can lipo this kind of output. If not (and it is an actual
    // output) then we disallow, since we can't create an output file with the
    // right name without overwriting it. We could remove this oddity by just
    // changing the output names to include the arch, which would also fix
    // -save-temps. Compatibility wins for now.

    if (Archs.size() > 1 && !types::canLipoType(Act->getType()))
      Diag(clang::diag::err_drv_invalid_output_with_multiple_archs)
        << types::getTypeName(Act->getType());

    ActionList Inputs;
    for (unsigned i = 0, e = Archs.size(); i != e; ++i) {
      Inputs.push_back(new BindArchAction(Act, Archs[i]));
      if (i != 0)
        Inputs.back()->setOwnsInputs(false);
    }

    // Lipo if necessary, we do it this way because we need to set the arch flag
    // so that -Xarch_ gets overwritten.
    if (Inputs.size() == 1 || Act->getType() == types::TY_Nothing)
      Actions.append(Inputs.begin(), Inputs.end());
    else
      Actions.push_back(new LipoJobAction(Inputs, Act->getType()));

    // Handle debug info queries.
    Arg *A = Args.getLastArg(options::OPT_g_Group);
    if (A && !A->getOption().matches(options::OPT_g0) &&
        !A->getOption().matches(options::OPT_gstabs) &&
        ContainsCompileOrAssembleAction(Actions.back())) {

      // Add a 'dsymutil' step if necessary, when debug info is enabled and we
      // have a compile input. We need to run 'dsymutil' ourselves in such cases
      // because the debug info will refer to a temporary object file which is
      // will be removed at the end of the compilation process.
      if (Act->getType() == types::TY_Image) {
        ActionList Inputs;
        Inputs.push_back(Actions.back());
        Actions.pop_back();
        Actions.push_back(new DsymutilJobAction(Inputs, types::TY_dSYM));
      }

      // Verify the output (debug information only) if we passed '-verify'.
      if (Args.hasArg(options::OPT_verify)) {
        ActionList VerifyInputs;
        VerifyInputs.push_back(Actions.back());
        Actions.pop_back();
        Actions.push_back(new VerifyJobAction(VerifyInputs,
                                              types::TY_Nothing));
      }
    }
  }
}

// Construct a the list of inputs and their types.
void Driver::BuildInputs(const ToolChain &TC, const DerivedArgList &Args,
                         InputList &Inputs) const {
  // Track the current user specified (-x) input. We also explicitly track the
  // argument used to set the type; we only want to claim the type when we
  // actually use it, so we warn about unused -x arguments.
  types::ID InputType = types::TY_Nothing;
  Arg *InputTypeArg = 0;

  for (ArgList::const_iterator it = Args.begin(), ie = Args.end();
       it != ie; ++it) {
    Arg *A = *it;

    if (A->getOption().getKind() == Option::InputClass) {
      const char *Value = A->getValue();
      types::ID Ty = types::TY_INVALID;

      // Infer the input type if necessary.
      if (InputType == types::TY_Nothing) {
        // If there was an explicit arg for this, claim it.
        if (InputTypeArg)
          InputTypeArg->claim();

        // stdin must be handled specially.
        if (memcmp(Value, "-", 2) == 0) {
          // If running with -E, treat as a C input (this changes the builtin
          // macros, for example). This may be overridden by -ObjC below.
          //
          // Otherwise emit an error but still use a valid type to avoid
          // spurious errors (e.g., no inputs).
          if (!Args.hasArgNoClaim(options::OPT_E) && !CCCIsCPP)
            Diag(clang::diag::err_drv_unknown_stdin_type);
          Ty = types::TY_C;
        } else {
          // Otherwise lookup by extension.
          // Fallback is C if invoked as C preprocessor or Object otherwise.
          // We use a host hook here because Darwin at least has its own
          // idea of what .s is.
          if (const char *Ext = strrchr(Value, '.'))
            Ty = TC.LookupTypeForExtension(Ext + 1);

          if (Ty == types::TY_INVALID) {
            if (CCCIsCPP)
              Ty = types::TY_C;
            else
              Ty = types::TY_Object;
          }

          // If the driver is invoked as C++ compiler (like clang++ or c++) it
          // should autodetect some input files as C++ for g++ compatibility.
          if (CCCIsCXX) {
            types::ID OldTy = Ty;
            Ty = types::lookupCXXTypeForCType(Ty);

            if (Ty != OldTy)
              Diag(clang::diag::warn_drv_treating_input_as_cxx)
                << getTypeName(OldTy) << getTypeName(Ty);
          }
        }

        // -ObjC and -ObjC++ override the default language, but only for "source
        // files". We just treat everything that isn't a linker input as a
        // source file.
        //
        // FIXME: Clean this up if we move the phase sequence into the type.
        if (Ty != types::TY_Object) {
          if (Args.hasArg(options::OPT_ObjC))
            Ty = types::TY_ObjC;
          else if (Args.hasArg(options::OPT_ObjCXX))
            Ty = types::TY_ObjCXX;
        }
      } else {
        assert(InputTypeArg && "InputType set w/o InputTypeArg");
        InputTypeArg->claim();
        Ty = InputType;
      }

      // Check that the file exists, if enabled.
      if (CheckInputsExist && memcmp(Value, "-", 2) != 0) {
        SmallString<64> Path(Value);
        if (Arg *WorkDir = Args.getLastArg(options::OPT_working_directory)) {
          if (!llvm::sys::path::is_absolute(Path.str())) {
            SmallString<64> Directory(WorkDir->getValue());
            llvm::sys::path::append(Directory, Value);
            Path.assign(Directory);
          }
        }

        bool exists = false;
        if (llvm::sys::fs::exists(Path.c_str(), exists) || !exists)
          Diag(clang::diag::err_drv_no_such_file) << Path.str();
        else
          Inputs.push_back(std::make_pair(Ty, A));
      } else
        Inputs.push_back(std::make_pair(Ty, A));

    } else if (A->getOption().hasFlag(options::LinkerInput)) {
      // Just treat as object type, we could make a special type for this if
      // necessary.
      Inputs.push_back(std::make_pair(types::TY_Object, A));

    } else if (A->getOption().matches(options::OPT_x)) {
      InputTypeArg = A;
      InputType = types::lookupTypeForTypeSpecifier(A->getValue());
      A->claim();

      // Follow gcc behavior and treat as linker input for invalid -x
      // options. Its not clear why we shouldn't just revert to unknown; but
      // this isn't very important, we might as well be bug compatible.
      if (!InputType) {
        Diag(clang::diag::err_drv_unknown_language) << A->getValue();
        InputType = types::TY_Object;
      }
    }
  }
  if (CCCIsCPP && Inputs.empty()) {
    // If called as standalone preprocessor, stdin is processed
    // if no other input is present.
    unsigned Index = Args.getBaseArgs().MakeIndex("-");
    Arg *A = Opts->ParseOneArg(Args, Index);
    A->claim();
    Inputs.push_back(std::make_pair(types::TY_C, A));
  }
}

void Driver::BuildActions(const ToolChain &TC, const DerivedArgList &Args,
                          const InputList &Inputs, ActionList &Actions) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation actions");

  if (!SuppressMissingInputWarning && Inputs.empty()) {
    Diag(clang::diag::err_drv_no_input_files);
    return;
  }

  Arg *FinalPhaseArg;
  phases::ID FinalPhase = getFinalPhase(Args, &FinalPhaseArg);

  // Reject -Z* at the top level, these options should never have been exposed
  // by gcc.
  if (Arg *A = Args.getLastArg(options::OPT_Z_Joined))
    Diag(clang::diag::err_drv_use_of_Z_option) << A->getAsString(Args);

  // Construct the actions to perform.
  ActionList LinkerInputs;
  unsigned NumSteps = 0;
  for (unsigned i = 0, e = Inputs.size(); i != e; ++i) {
    types::ID InputType = Inputs[i].first;
    const Arg *InputArg = Inputs[i].second;

    NumSteps = types::getNumCompilationPhases(InputType);
    assert(NumSteps && "Invalid number of steps!");

    // If the first step comes after the final phase we are doing as part of
    // this compilation, warn the user about it.
    phases::ID InitialPhase = types::getCompilationPhase(InputType, 0);
    if (InitialPhase > FinalPhase) {
      // Claim here to avoid the more general unused warning.
      InputArg->claim();

      // Suppress all unused style warnings with -Qunused-arguments
      if (Args.hasArg(options::OPT_Qunused_arguments))
        continue;

      // Special case when final phase determined by binary name, rather than
      // by a command-line argument with a corresponding Arg.
      if (CCCIsCPP)
        Diag(clang::diag::warn_drv_input_file_unused_by_cpp)
          << InputArg->getAsString(Args)
          << getPhaseName(InitialPhase);
      // Special case '-E' warning on a previously preprocessed file to make
      // more sense.
      else if (InitialPhase == phases::Compile &&
               FinalPhase == phases::Preprocess &&
               getPreprocessedType(InputType) == types::TY_INVALID)
        Diag(clang::diag::warn_drv_preprocessed_input_file_unused)
          << InputArg->getAsString(Args)
          << !!FinalPhaseArg
          << FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "";
      else
        Diag(clang::diag::warn_drv_input_file_unused)
          << InputArg->getAsString(Args)
          << getPhaseName(InitialPhase)
          << !!FinalPhaseArg
          << FinalPhaseArg ? FinalPhaseArg->getOption().getName() : "";
      continue;
    }

    // Build the pipeline for this file.
    OwningPtr<Action> Current(new InputAction(*InputArg, InputType));
    for (unsigned i = 0; i != NumSteps; ++i) {
      phases::ID Phase = types::getCompilationPhase(InputType, i);

      // We are done if this step is past what the user requested.
      if (Phase > FinalPhase)
        break;

      // Queue linker inputs.
      if (Phase == phases::Link) {
        assert(i + 1 == NumSteps && "linking must be final compilation step.");
        LinkerInputs.push_back(Current.take());
        break;
      }

      // Some types skip the assembler phase (e.g., llvm-bc), but we can't
      // encode this in the steps because the intermediate type depends on
      // arguments. Just special case here.
      if (Phase == phases::Assemble && Current->getType() != types::TY_PP_Asm)
        continue;

      // Otherwise construct the appropriate action.
      Current.reset(ConstructPhaseAction(Args, Phase, Current.take()));
      if (Current->getType() == types::TY_Nothing)
        break;
    }

    // If we ended with something, add to the output list.
    if (Current)
      Actions.push_back(Current.take());
  }

  // Add a link action if necessary.
  if (!LinkerInputs.empty())
    Actions.push_back(new LinkJobAction(LinkerInputs, types::TY_Image));

  // If we are linking, claim any options which are obviously only used for
  // compilation.
  if (FinalPhase == phases::Link && (NumSteps == 1))
    Args.ClaimAllArgs(options::OPT_CompileOnly_Group);
}

Action *Driver::ConstructPhaseAction(const ArgList &Args, phases::ID Phase,
                                     Action *Input) const {
  llvm::PrettyStackTraceString CrashInfo("Constructing phase actions");
  // Build the appropriate action.
  switch (Phase) {
  case phases::Link: llvm_unreachable("link action invalid here.");
  case phases::Preprocess: {
    types::ID OutputTy;
    // -{M, MM} alter the output type.
    if (Args.hasArg(options::OPT_M, options::OPT_MM)) {
      OutputTy = types::TY_Dependencies;
    } else {
      OutputTy = Input->getType();
      if (!Args.hasFlag(options::OPT_frewrite_includes,
                        options::OPT_fno_rewrite_includes, false))
        OutputTy = types::getPreprocessedType(OutputTy);
      assert(OutputTy != types::TY_INVALID &&
             "Cannot preprocess this input type!");
    }
    return new PreprocessJobAction(Input, OutputTy);
  }
  case phases::Precompile: {
    types::ID OutputTy = types::TY_PCH;
    if (Args.hasArg(options::OPT_fsyntax_only)) {
      // Syntax checks should not emit a PCH file
      OutputTy = types::TY_Nothing;
    }
    return new PrecompileJobAction(Input, OutputTy);
  }
  case phases::Compile: {
    if (Args.hasArg(options::OPT_fsyntax_only)) {
      return new CompileJobAction(Input, types::TY_Nothing);
    } else if (Args.hasArg(options::OPT_rewrite_objc)) {
      return new CompileJobAction(Input, types::TY_RewrittenObjC);
    } else if (Args.hasArg(options::OPT_rewrite_legacy_objc)) {
      return new CompileJobAction(Input, types::TY_RewrittenLegacyObjC);
    } else if (Args.hasArg(options::OPT__analyze, options::OPT__analyze_auto)) {
      return new AnalyzeJobAction(Input, types::TY_Plist);
    } else if (Args.hasArg(options::OPT__migrate)) {
      return new MigrateJobAction(Input, types::TY_Remap);
    } else if (Args.hasArg(options::OPT_emit_ast)) {
      return new CompileJobAction(Input, types::TY_AST);
    } else if (IsUsingLTO(Args)) {
      types::ID Output =
        Args.hasArg(options::OPT_S) ? types::TY_LTO_IR : types::TY_LTO_BC;
      return new CompileJobAction(Input, Output);
    } else {
      return new CompileJobAction(Input, types::TY_PP_Asm);
    }
  }
  case phases::Assemble:
    return new AssembleJobAction(Input, types::TY_Object);
  }

  llvm_unreachable("invalid phase in ConstructPhaseAction");
}

bool Driver::IsUsingLTO(const ArgList &Args) const {
  // Check for -emit-llvm or -flto.
  if (Args.hasArg(options::OPT_emit_llvm) ||
      Args.hasFlag(options::OPT_flto, options::OPT_fno_lto, false))
    return true;

  // Check for -O4.
  if (const Arg *A = Args.getLastArg(options::OPT_O_Group))
      return A->getOption().matches(options::OPT_O4);

  return false;
}

void Driver::BuildJobs(Compilation &C) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o);

  // It is an error to provide a -o option if we are making multiple output
  // files.
  if (FinalOutput) {
    unsigned NumOutputs = 0;
    for (ActionList::const_iterator it = C.getActions().begin(),
           ie = C.getActions().end(); it != ie; ++it)
      if ((*it)->getType() != types::TY_Nothing)
        ++NumOutputs;

    if (NumOutputs > 1) {
      Diag(clang::diag::err_drv_output_argument_with_multiple_files);
      FinalOutput = 0;
    }
  }

  for (ActionList::const_iterator it = C.getActions().begin(),
         ie = C.getActions().end(); it != ie; ++it) {
    Action *A = *it;

    // If we are linking an image for multiple archs then the linker wants
    // -arch_multiple and -final_output <final image name>. Unfortunately, this
    // doesn't fit in cleanly because we have to pass this information down.
    //
    // FIXME: This is a hack; find a cleaner way to integrate this into the
    // process.
    const char *LinkingOutput = 0;
    if (isa<LipoJobAction>(A)) {
      if (FinalOutput)
        LinkingOutput = FinalOutput->getValue();
      else
        LinkingOutput = DefaultImageName.c_str();
    }

    InputInfo II;
    BuildJobsForAction(C, A, &C.getDefaultToolChain(),
                       /*BoundArch*/0,
                       /*AtTopLevel*/ true,
                       /*LinkingOutput*/ LinkingOutput,
                       II);
  }

  // If the user passed -Qunused-arguments or there were errors, don't warn
  // about any unused arguments.
  if (Diags.hasErrorOccurred() ||
      C.getArgs().hasArg(options::OPT_Qunused_arguments))
    return;

  // Claim -### here.
  (void) C.getArgs().hasArg(options::OPT__HASH_HASH_HASH);

  for (ArgList::const_iterator it = C.getArgs().begin(), ie = C.getArgs().end();
       it != ie; ++it) {
    Arg *A = *it;

    // FIXME: It would be nice to be able to send the argument to the
    // DiagnosticsEngine, so that extra values, position, and so on could be
    // printed.
    if (!A->isClaimed()) {
      if (A->getOption().hasFlag(options::NoArgumentUnused))
        continue;

      // Suppress the warning automatically if this is just a flag, and it is an
      // instance of an argument we already claimed.
      const Option &Opt = A->getOption();
      if (Opt.getKind() == Option::FlagClass) {
        bool DuplicateClaimed = false;

        for (arg_iterator it = C.getArgs().filtered_begin(&Opt),
               ie = C.getArgs().filtered_end(); it != ie; ++it) {
          if ((*it)->isClaimed()) {
            DuplicateClaimed = true;
            break;
          }
        }

        if (DuplicateClaimed)
          continue;
      }

      Diag(clang::diag::warn_drv_unused_argument)
        << A->getAsString(C.getArgs());
    }
  }
}

static const Tool &SelectToolForJob(Compilation &C, const ToolChain *TC,
                                    const JobAction *JA,
                                    const ActionList *&Inputs) {
  const Tool *ToolForJob = 0;

  // See if we should look for a compiler with an integrated assembler. We match
  // bottom up, so what we are actually looking for is an assembler job with a
  // compiler input.

  if (C.getArgs().hasFlag(options::OPT_integrated_as,
                          options::OPT_no_integrated_as,
                          TC->IsIntegratedAssemblerDefault()) &&
      !C.getArgs().hasArg(options::OPT_save_temps) &&
      isa<AssembleJobAction>(JA) &&
      Inputs->size() == 1 && isa<CompileJobAction>(*Inputs->begin())) {
    const Tool &Compiler = TC->SelectTool(
      C, cast<JobAction>(**Inputs->begin()), (*Inputs)[0]->getInputs());
    if (Compiler.hasIntegratedAssembler()) {
      Inputs = &(*Inputs)[0]->getInputs();
      ToolForJob = &Compiler;
    }
  }

  // Otherwise use the tool for the current job.
  if (!ToolForJob)
    ToolForJob = &TC->SelectTool(C, *JA, *Inputs);

  // See if we should use an integrated preprocessor. We do so when we have
  // exactly one input, since this is the only use case we care about
  // (irrelevant since we don't support combine yet).
  if (Inputs->size() == 1 && isa<PreprocessJobAction>(*Inputs->begin()) &&
      !C.getArgs().hasArg(options::OPT_no_integrated_cpp) &&
      !C.getArgs().hasArg(options::OPT_traditional_cpp) &&
      !C.getArgs().hasArg(options::OPT_save_temps) &&
      !C.getArgs().hasArg(options::OPT_rewrite_objc) &&
      ToolForJob->hasIntegratedCPP())
    Inputs = &(*Inputs)[0]->getInputs();

  return *ToolForJob;
}

void Driver::BuildJobsForAction(Compilation &C,
                                const Action *A,
                                const ToolChain *TC,
                                const char *BoundArch,
                                bool AtTopLevel,
                                const char *LinkingOutput,
                                InputInfo &Result) const {
  llvm::PrettyStackTraceString CrashInfo("Building compilation jobs");

  if (const InputAction *IA = dyn_cast<InputAction>(A)) {
    // FIXME: It would be nice to not claim this here; maybe the old scheme of
    // just using Args was better?
    const Arg &Input = IA->getInputArg();
    Input.claim();
    if (Input.getOption().matches(options::OPT_INPUT)) {
      const char *Name = Input.getValue();
      Result = InputInfo(Name, A->getType(), Name);
    } else
      Result = InputInfo(&Input, A->getType(), "");
    return;
  }

  if (const BindArchAction *BAA = dyn_cast<BindArchAction>(A)) {
    const ToolChain *TC;
    const char *ArchName = BAA->getArchName();

    if (ArchName)
      TC = &getToolChain(C.getArgs(), ArchName);
    else
      TC = &C.getDefaultToolChain();

    BuildJobsForAction(C, *BAA->begin(), TC, BAA->getArchName(),
                       AtTopLevel, LinkingOutput, Result);
    return;
  }

  const ActionList *Inputs = &A->getInputs();

  const JobAction *JA = cast<JobAction>(A);
  const Tool &T = SelectToolForJob(C, TC, JA, Inputs);

  // Only use pipes when there is exactly one input.
  InputInfoList InputInfos;
  for (ActionList::const_iterator it = Inputs->begin(), ie = Inputs->end();
       it != ie; ++it) {
    // Treat dsymutil sub-jobs as being at the top-level too, they shouldn't get
    // temporary output names.
    //
    // FIXME: Clean this up.
    bool SubJobAtTopLevel = false;
    if (AtTopLevel && isa<DsymutilJobAction>(A))
      SubJobAtTopLevel = true;

    // Also treat verify sub-jobs as being at the top-level. They don't
    // produce any output and so don't need temporary output names.
    if (AtTopLevel && isa<VerifyJobAction>(A))
      SubJobAtTopLevel = true;

    InputInfo II;
    BuildJobsForAction(C, *it, TC, BoundArch,
                       SubJobAtTopLevel, LinkingOutput, II);
    InputInfos.push_back(II);
  }

  // Always use the first input as the base input.
  const char *BaseInput = InputInfos[0].getBaseInput();

  // ... except dsymutil actions, which use their actual input as the base
  // input.
  if (JA->getType() == types::TY_dSYM)
    BaseInput = InputInfos[0].getFilename();

  // Determine the place to write output to, if any.
  if (JA->getType() == types::TY_Nothing) {
    Result = InputInfo(A->getType(), BaseInput);
  } else {
    Result = InputInfo(GetNamedOutputPath(C, *JA, BaseInput, AtTopLevel),
                       A->getType(), BaseInput);
  }

  if (CCCPrintBindings && !CCGenDiagnostics) {
    llvm::errs() << "# \"" << T.getToolChain().getTripleString() << '"'
                 << " - \"" << T.getName() << "\", inputs: [";
    for (unsigned i = 0, e = InputInfos.size(); i != e; ++i) {
      llvm::errs() << InputInfos[i].getAsString();
      if (i + 1 != e)
        llvm::errs() << ", ";
    }
    llvm::errs() << "], output: " << Result.getAsString() << "\n";
  } else {
    T.ConstructJob(C, *JA, Result, InputInfos,
                   C.getArgsForToolChain(TC, BoundArch), LinkingOutput);
  }
}

const char *Driver::GetNamedOutputPath(Compilation &C,
                                       const JobAction &JA,
                                       const char *BaseInput,
                                       bool AtTopLevel) const {
  llvm::PrettyStackTraceString CrashInfo("Computing output path");
  // Output to a user requested destination?
  if (AtTopLevel && !isa<DsymutilJobAction>(JA) &&
      !isa<VerifyJobAction>(JA)) {
    if (Arg *FinalOutput = C.getArgs().getLastArg(options::OPT_o))
      return C.addResultFile(FinalOutput->getValue());
  }

  // Default to writing to stdout?
  if (AtTopLevel && isa<PreprocessJobAction>(JA) && !CCGenDiagnostics)
    return "-";

  // Output to a temporary file?
  if ((!AtTopLevel && !C.getArgs().hasArg(options::OPT_save_temps)) ||
      CCGenDiagnostics) {
    StringRef Name = llvm::sys::path::filename(BaseInput);
    std::pair<StringRef, StringRef> Split = Name.split('.');
    std::string TmpName =
      GetTemporaryPath(Split.first, types::getTypeTempSuffix(JA.getType()));
    return C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));
  }

  SmallString<128> BasePath(BaseInput);
  StringRef BaseName;

  // Dsymutil actions should use the full path.
  if (isa<DsymutilJobAction>(JA) || isa<VerifyJobAction>(JA))
    BaseName = BasePath;
  else
    BaseName = llvm::sys::path::filename(BasePath);

  // Determine what the derived output name should be.
  const char *NamedOutput;
  if (JA.getType() == types::TY_Image) {
    NamedOutput = DefaultImageName.c_str();
  } else {
    const char *Suffix = types::getTypeTempSuffix(JA.getType());
    assert(Suffix && "All types used for output should have a suffix.");

    std::string::size_type End = std::string::npos;
    if (!types::appendSuffixForType(JA.getType()))
      End = BaseName.rfind('.');
    std::string Suffixed(BaseName.substr(0, End));
    Suffixed += '.';
    Suffixed += Suffix;
    NamedOutput = C.getArgs().MakeArgString(Suffixed.c_str());
  }

  // If we're saving temps and the temp file conflicts with the input file,
  // then avoid overwriting input file.
  if (!AtTopLevel && C.getArgs().hasArg(options::OPT_save_temps) &&
      NamedOutput == BaseName) {

    bool SameFile = false;
    SmallString<256> Result;
    llvm::sys::fs::current_path(Result);
    llvm::sys::path::append(Result, BaseName);
    llvm::sys::fs::equivalent(BaseInput, Result.c_str(), SameFile);
    // Must share the same path to conflict.
    if (SameFile) {
      StringRef Name = llvm::sys::path::filename(BaseInput);
      std::pair<StringRef, StringRef> Split = Name.split('.');
      std::string TmpName =
        GetTemporaryPath(Split.first, types::getTypeTempSuffix(JA.getType()));
      return C.addTempFile(C.getArgs().MakeArgString(TmpName.c_str()));
    }
  }

  // As an annoying special case, PCH generation doesn't strip the pathname.
  if (JA.getType() == types::TY_PCH) {
    llvm::sys::path::remove_filename(BasePath);
    if (BasePath.empty())
      BasePath = NamedOutput;
    else
      llvm::sys::path::append(BasePath, NamedOutput);
    return C.addResultFile(C.getArgs().MakeArgString(BasePath.c_str()));
  } else {
    return C.addResultFile(NamedOutput);
  }
}

std::string Driver::GetFilePath(const char *Name, const ToolChain &TC) const {
  // Respect a limited subset of the '-Bprefix' functionality in GCC by
  // attempting to use this prefix when looking for file paths.
  for (Driver::prefix_list::const_iterator it = PrefixDirs.begin(),
       ie = PrefixDirs.end(); it != ie; ++it) {
    std::string Dir(*it);
    if (Dir.empty())
      continue;
    if (Dir[0] == '=')
      Dir = SysRoot + Dir.substr(1);
    llvm::sys::Path P(Dir);
    P.appendComponent(Name);
    bool Exists;
    if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
      return P.str();
  }

  llvm::sys::Path P(ResourceDir);
  P.appendComponent(Name);
  bool Exists;
  if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
    return P.str();

  const ToolChain::path_list &List = TC.getFilePaths();
  for (ToolChain::path_list::const_iterator
         it = List.begin(), ie = List.end(); it != ie; ++it) {
    std::string Dir(*it);
    if (Dir.empty())
      continue;
    if (Dir[0] == '=')
      Dir = SysRoot + Dir.substr(1);
    llvm::sys::Path P(Dir);
    P.appendComponent(Name);
    bool Exists;
    if (!llvm::sys::fs::exists(P.str(), Exists) && Exists)
      return P.str();
  }

  return Name;
}

std::string Driver::GetProgramPath(const char *Name,
                                   const ToolChain &TC) const {
  // FIXME: Needs a better variable than DefaultTargetTriple
  std::string TargetSpecificExecutable(DefaultTargetTriple + "-" + Name);
  // Respect a limited subset of the '-Bprefix' functionality in GCC by
  // attempting to use this prefix when looking for program paths.
  for (Driver::prefix_list::const_iterator it = PrefixDirs.begin(),
       ie = PrefixDirs.end(); it != ie; ++it) {
    bool IsDirectory;
    if (!llvm::sys::fs::is_directory(*it, IsDirectory) && IsDirectory) {
      llvm::sys::Path P(*it);
      P.appendComponent(TargetSpecificExecutable);
      if (P.canExecute()) return P.str();
      P.eraseComponent();
      P.appendComponent(Name);
      if (P.canExecute()) return P.str();
    } else {
      llvm::sys::Path P(*it + Name);
      if (P.canExecute()) return P.str();
    }
  }

  const ToolChain::path_list &List = TC.getProgramPaths();
  for (ToolChain::path_list::const_iterator
         it = List.begin(), ie = List.end(); it != ie; ++it) {
    llvm::sys::Path P(*it);
    P.appendComponent(TargetSpecificExecutable);
    if (P.canExecute()) return P.str();
    P.eraseComponent();
    P.appendComponent(Name);
    if (P.canExecute()) return P.str();
  }

  // If all else failed, search the path.
  llvm::sys::Path
      P(llvm::sys::Program::FindProgramByName(TargetSpecificExecutable));
  if (!P.empty())
    return P.str();

  P = llvm::sys::Path(llvm::sys::Program::FindProgramByName(Name));
  if (!P.empty())
    return P.str();

  return Name;
}

std::string Driver::GetTemporaryPath(StringRef Prefix, const char *Suffix)
  const {
  // FIXME: This is lame; sys::Path should provide this function (in particular,
  // it should know how to find the temporary files dir).
  std::string Error;
  const char *TmpDir = ::getenv("TMPDIR");
  if (!TmpDir)
    TmpDir = ::getenv("TEMP");
  if (!TmpDir)
    TmpDir = ::getenv("TMP");
  if (!TmpDir)
    TmpDir = "/tmp";
  llvm::sys::Path P(TmpDir);
  P.appendComponent(Prefix);
  if (P.makeUnique(false, &Error)) {
    Diag(clang::diag::err_unable_to_make_temp) << Error;
    return "";
  }

  // FIXME: Grumble, makeUnique sometimes leaves the file around!?  PR3837.
  P.eraseFromDisk(false, 0);

  if (Suffix)
    P.appendSuffix(Suffix);
  return P.str();
}

/// \brief Compute target triple from args.
///
/// This routine provides the logic to compute a target triple from various
/// args passed to the driver and the default triple string.
static llvm::Triple computeTargetTriple(StringRef DefaultTargetTriple,
                                        const ArgList &Args,
                                        StringRef DarwinArchName) {
  // FIXME: Already done in Compilation *Driver::BuildCompilation
  if (const Arg *A = Args.getLastArg(options::OPT_target))
    DefaultTargetTriple = A->getValue();

  llvm::Triple Target(llvm::Triple::normalize(DefaultTargetTriple));

  // Handle Darwin-specific options available here.
  if (Target.isOSDarwin()) {
    // If an explict Darwin arch name is given, that trumps all.
    if (!DarwinArchName.empty()) {
      Target.setArch(
        tools::darwin::getArchTypeForDarwinArchName(DarwinArchName));
      return Target;
    }

    // Handle the Darwin '-arch' flag.
    if (Arg *A = Args.getLastArg(options::OPT_arch)) {
      llvm::Triple::ArchType DarwinArch
        = tools::darwin::getArchTypeForDarwinArchName(A->getValue());
      if (DarwinArch != llvm::Triple::UnknownArch)
        Target.setArch(DarwinArch);
    }
  }

  // Skip further flag support on OSes which don't support '-m32' or '-m64'.
  if (Target.getArchName() == "tce" ||
      Target.getOS() == llvm::Triple::AuroraUX ||
      Target.getOS() == llvm::Triple::Minix)
    return Target;

  // Handle pseudo-target flags '-m32' and '-m64'.
  // FIXME: Should this information be in llvm::Triple?
  if (Arg *A = Args.getLastArg(options::OPT_m32, options::OPT_m64)) {
    if (A->getOption().matches(options::OPT_m32)) {
      if (Target.getArch() == llvm::Triple::x86_64)
        Target.setArch(llvm::Triple::x86);
      if (Target.getArch() == llvm::Triple::ppc64)
        Target.setArch(llvm::Triple::ppc);
    } else {
      if (Target.getArch() == llvm::Triple::x86)
        Target.setArch(llvm::Triple::x86_64);
      if (Target.getArch() == llvm::Triple::ppc)
        Target.setArch(llvm::Triple::ppc64);
    }
  }

  return Target;
}

const ToolChain &Driver::getToolChain(const ArgList &Args,
                                      StringRef DarwinArchName) const {
  llvm::Triple Target = computeTargetTriple(DefaultTargetTriple, Args,
                                            DarwinArchName);

  ToolChain *&TC = ToolChains[Target.str()];
  if (!TC) {
    switch (Target.getOS()) {
    case llvm::Triple::AuroraUX:
      TC = new toolchains::AuroraUX(*this, Target, Args);
      break;
    case llvm::Triple::Darwin:
    case llvm::Triple::MacOSX:
    case llvm::Triple::IOS:
      if (Target.getArch() == llvm::Triple::x86 ||
          Target.getArch() == llvm::Triple::x86_64 ||
          Target.getArch() == llvm::Triple::arm ||
          Target.getArch() == llvm::Triple::thumb)
        TC = new toolchains::DarwinClang(*this, Target);
      else
        TC = new toolchains::Darwin_Generic_GCC(*this, Target, Args);
      break;
    case llvm::Triple::DragonFly:
      TC = new toolchains::DragonFly(*this, Target, Args);
      break;
    case llvm::Triple::OpenBSD:
      TC = new toolchains::OpenBSD(*this, Target, Args);
      break;
    case llvm::Triple::Bitrig:
      TC = new toolchains::Bitrig(*this, Target, Args);
      break;
    case llvm::Triple::NetBSD:
      TC = new toolchains::NetBSD(*this, Target, Args);
      break;
    case llvm::Triple::FreeBSD:
      TC = new toolchains::FreeBSD(*this, Target, Args);
      break;
    case llvm::Triple::Minix:
      TC = new toolchains::Minix(*this, Target, Args);
      break;
    case llvm::Triple::Linux:
      if (Target.getArch() == llvm::Triple::hexagon)
        TC = new toolchains::Hexagon_TC(*this, Target);
      else
        TC = new toolchains::Linux(*this, Target, Args);
      break;
    case llvm::Triple::Solaris:
      TC = new toolchains::Solaris(*this, Target, Args);
      break;
    case llvm::Triple::Win32:
      TC = new toolchains::Windows(*this, Target);
      break;
    case llvm::Triple::MinGW32:
      // FIXME: We need a MinGW toolchain. Fallthrough for now.
    default:
      // TCE is an OSless target
      if (Target.getArchName() == "tce") {
        TC = new toolchains::TCEToolChain(*this, Target);
        break;
      }

      TC = new toolchains::Generic_GCC(*this, Target, Args);
      break;
    }
  }
  return *TC;
}

bool Driver::ShouldUseClangCompiler(const Compilation &C, const JobAction &JA,
                                    const llvm::Triple &Triple) const {
  // Check if user requested no clang, or clang doesn't understand this type (we
  // only handle single inputs for now).
  if (JA.size() != 1 ||
      !types::isAcceptedByClang((*JA.begin())->getType()))
    return false;

  // Otherwise make sure this is an action clang understands.
  if (!isa<PreprocessJobAction>(JA) && !isa<PrecompileJobAction>(JA) &&
      !isa<CompileJobAction>(JA))
    return false;

  return true;
}

/// GetReleaseVersion - Parse (([0-9]+)(.([0-9]+)(.([0-9]+)?))?)? and return the
/// grouped values as integers. Numbers which are not provided are set to 0.
///
/// \return True if the entire string was parsed (9.2), or all groups were
/// parsed (10.3.5extrastuff).
bool Driver::GetReleaseVersion(const char *Str, unsigned &Major,
                               unsigned &Minor, unsigned &Micro,
                               bool &HadExtra) {
  HadExtra = false;

  Major = Minor = Micro = 0;
  if (*Str == '\0')
    return true;

  char *End;
  Major = (unsigned) strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;

  Str = End+1;
  Minor = (unsigned) strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (*End != '.')
    return false;

  Str = End+1;
  Micro = (unsigned) strtol(Str, &End, 10);
  if (*Str != '\0' && *End == '\0')
    return true;
  if (Str == End)
    return false;
  HadExtra = true;
  return true;
}
