//===- DriverUtils.cpp ----------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains utility functions for the driver. Because there
// are so many small functions, we created this separate file to make
// Driver.cpp less cluttered.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "Symbols.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace llvm::COFF;
using namespace llvm;
using llvm::cl::ExpandResponseFiles;
using llvm::cl::TokenizeWindowsCommandLine;
using llvm::sys::Process;

namespace lld {
namespace coff {
namespace {

class Executor {
public:
  explicit Executor(StringRef S) : Saver(Alloc), Prog(Saver.save(S)) {}
  void add(StringRef S)    { Args.push_back(Saver.save(S)); }
  void add(std::string &S) { Args.push_back(Saver.save(S)); }
  void add(Twine S)        { Args.push_back(Saver.save(S)); }
  void add(const char *S)  { Args.push_back(Saver.save(S)); }

  void run() {
    ErrorOr<std::string> ExeOrErr = llvm::sys::findProgramByName(Prog);
    if (auto EC = ExeOrErr.getError())
      fatal(EC, "unable to find " + Prog + " in PATH: ");
    const char *Exe = Saver.save(*ExeOrErr);
    Args.insert(Args.begin(), Exe);
    Args.push_back(nullptr);
    if (llvm::sys::ExecuteAndWait(Args[0], Args.data()) != 0) {
      for (const char *S : Args)
        if (S)
          llvm::errs() << S << " ";
      fatal("ExecuteAndWait failed");
    }
  }

private:
  llvm::BumpPtrAllocator Alloc;
  llvm::StringSaver Saver;
  StringRef Prog;
  std::vector<const char *> Args;
};

} // anonymous namespace

// Returns /machine's value.
MachineTypes getMachineType(StringRef S) {
  MachineTypes MT = StringSwitch<MachineTypes>(S.lower())
                        .Case("x64", AMD64)
                        .Case("amd64", AMD64)
                        .Case("x86", I386)
                        .Case("i386", I386)
                        .Case("arm", ARMNT)
                        .Default(IMAGE_FILE_MACHINE_UNKNOWN);
  if (MT != IMAGE_FILE_MACHINE_UNKNOWN)
    return MT;
  fatal("unknown /machine argument: " + S);
}

StringRef machineToStr(MachineTypes MT) {
  switch (MT) {
  case ARMNT:
    return "arm";
  case AMD64:
    return "x64";
  case I386:
    return "x86";
  default:
    llvm_unreachable("unknown machine type");
  }
}

// Parses a string in the form of "<integer>[,<integer>]".
void parseNumbers(StringRef Arg, uint64_t *Addr, uint64_t *Size) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split(',');
  if (S1.getAsInteger(0, *Addr))
    fatal("invalid number: " + S1);
  if (Size && !S2.empty() && S2.getAsInteger(0, *Size))
    fatal("invalid number: " + S2);
}

// Parses a string in the form of "<integer>[.<integer>]".
// If second number is not present, Minor is set to 0.
void parseVersion(StringRef Arg, uint32_t *Major, uint32_t *Minor) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split('.');
  if (S1.getAsInteger(0, *Major))
    fatal("invalid number: " + S1);
  *Minor = 0;
  if (!S2.empty() && S2.getAsInteger(0, *Minor))
    fatal("invalid number: " + S2);
}

// Parses a string in the form of "<subsystem>[,<integer>[.<integer>]]".
void parseSubsystem(StringRef Arg, WindowsSubsystem *Sys, uint32_t *Major,
                    uint32_t *Minor) {
  StringRef SysStr, Ver;
  std::tie(SysStr, Ver) = Arg.split(',');
  *Sys = StringSwitch<WindowsSubsystem>(SysStr.lower())
    .Case("boot_application", IMAGE_SUBSYSTEM_WINDOWS_BOOT_APPLICATION)
    .Case("console", IMAGE_SUBSYSTEM_WINDOWS_CUI)
    .Case("efi_application", IMAGE_SUBSYSTEM_EFI_APPLICATION)
    .Case("efi_boot_service_driver", IMAGE_SUBSYSTEM_EFI_BOOT_SERVICE_DRIVER)
    .Case("efi_rom", IMAGE_SUBSYSTEM_EFI_ROM)
    .Case("efi_runtime_driver", IMAGE_SUBSYSTEM_EFI_RUNTIME_DRIVER)
    .Case("native", IMAGE_SUBSYSTEM_NATIVE)
    .Case("posix", IMAGE_SUBSYSTEM_POSIX_CUI)
    .Case("windows", IMAGE_SUBSYSTEM_WINDOWS_GUI)
    .Default(IMAGE_SUBSYSTEM_UNKNOWN);
  if (*Sys == IMAGE_SUBSYSTEM_UNKNOWN)
    fatal("unknown subsystem: " + SysStr);
  if (!Ver.empty())
    parseVersion(Ver, Major, Minor);
}

// Parse a string of the form of "<from>=<to>".
// Results are directly written to Config.
void parseAlternateName(StringRef S) {
  StringRef From, To;
  std::tie(From, To) = S.split('=');
  if (From.empty() || To.empty())
    fatal("/alternatename: invalid argument: " + S);
  auto It = Config->AlternateNames.find(From);
  if (It != Config->AlternateNames.end() && It->second != To)
    fatal("/alternatename: conflicts: " + S);
  Config->AlternateNames.insert(It, std::make_pair(From, To));
}

// Parse a string of the form of "<from>=<to>".
// Results are directly written to Config.
void parseMerge(StringRef S) {
  StringRef From, To;
  std::tie(From, To) = S.split('=');
  if (From.empty() || To.empty())
    fatal("/merge: invalid argument: " + S);
  auto Pair = Config->Merge.insert(std::make_pair(From, To));
  bool Inserted = Pair.second;
  if (!Inserted) {
    StringRef Existing = Pair.first->second;
    if (Existing != To)
      llvm::errs() << "warning: " << S << ": already merged into "
                   << Existing << "\n";
  }
}

static uint32_t parseSectionAttributes(StringRef S) {
  uint32_t Ret = 0;
  for (char C : S.lower()) {
    switch (C) {
    case 'd':
      Ret |= IMAGE_SCN_MEM_DISCARDABLE;
      break;
    case 'e':
      Ret |= IMAGE_SCN_MEM_EXECUTE;
      break;
    case 'k':
      Ret |= IMAGE_SCN_MEM_NOT_CACHED;
      break;
    case 'p':
      Ret |= IMAGE_SCN_MEM_NOT_PAGED;
      break;
    case 'r':
      Ret |= IMAGE_SCN_MEM_READ;
      break;
    case 's':
      Ret |= IMAGE_SCN_MEM_SHARED;
      break;
    case 'w':
      Ret |= IMAGE_SCN_MEM_WRITE;
      break;
    default:
      fatal("/section: invalid argument: " + S);
    }
  }
  return Ret;
}

// Parses /section option argument.
void parseSection(StringRef S) {
  StringRef Name, Attrs;
  std::tie(Name, Attrs) = S.split(',');
  if (Name.empty() || Attrs.empty())
    fatal("/section: invalid argument: " + S);
  Config->Section[Name] = parseSectionAttributes(Attrs);
}

// Parses a string in the form of "EMBED[,=<integer>]|NO".
// Results are directly written to Config.
void parseManifest(StringRef Arg) {
  if (Arg.equals_lower("no")) {
    Config->Manifest = Configuration::No;
    return;
  }
  if (!Arg.startswith_lower("embed"))
    fatal("invalid option " + Arg);
  Config->Manifest = Configuration::Embed;
  Arg = Arg.substr(strlen("embed"));
  if (Arg.empty())
    return;
  if (!Arg.startswith_lower(",id="))
    fatal("invalid option " + Arg);
  Arg = Arg.substr(strlen(",id="));
  if (Arg.getAsInteger(0, Config->ManifestID))
    fatal("invalid option " + Arg);
}

// Parses a string in the form of "level=<string>|uiAccess=<string>|NO".
// Results are directly written to Config.
void parseManifestUAC(StringRef Arg) {
  if (Arg.equals_lower("no")) {
    Config->ManifestUAC = false;
    return;
  }
  for (;;) {
    Arg = Arg.ltrim();
    if (Arg.empty())
      return;
    if (Arg.startswith_lower("level=")) {
      Arg = Arg.substr(strlen("level="));
      std::tie(Config->ManifestLevel, Arg) = Arg.split(" ");
      continue;
    }
    if (Arg.startswith_lower("uiaccess=")) {
      Arg = Arg.substr(strlen("uiaccess="));
      std::tie(Config->ManifestUIAccess, Arg) = Arg.split(" ");
      continue;
    }
    fatal("invalid option " + Arg);
  }
}

// Quote each line with "". Existing double-quote is converted
// to two double-quotes.
static void quoteAndPrint(raw_ostream &Out, StringRef S) {
  while (!S.empty()) {
    StringRef Line;
    std::tie(Line, S) = S.split("\n");
    if (Line.empty())
      continue;
    Out << '\"';
    for (int I = 0, E = Line.size(); I != E; ++I) {
      if (Line[I] == '\"') {
        Out << "\"\"";
      } else {
        Out << Line[I];
      }
    }
    Out << "\"\n";
  }
}

// An RAII temporary file class that automatically removes a temporary file.
namespace {
class TemporaryFile {
public:
  TemporaryFile(StringRef Prefix, StringRef Extn) {
    SmallString<128> S;
    if (auto EC = sys::fs::createTemporaryFile("lld-" + Prefix, Extn, S))
      fatal(EC, "cannot create a temporary file");
    Path = S.str();
  }

  TemporaryFile(TemporaryFile &&Obj) {
    std::swap(Path, Obj.Path);
  }

  ~TemporaryFile() {
    if (Path.empty())
      return;
    if (sys::fs::remove(Path))
      fatal("failed to remove " + Path);
  }

  // Returns a memory buffer of this temporary file.
  // Note that this function does not leave the file open,
  // so it is safe to remove the file immediately after this function
  // is called (you cannot remove an opened file on Windows.)
  std::unique_ptr<MemoryBuffer> getMemoryBuffer() {
    // IsVolatileSize=true forces MemoryBuffer to not use mmap().
    return check(MemoryBuffer::getFile(Path, /*FileSize=*/-1,
                                       /*RequiresNullTerminator=*/false,
                                       /*IsVolatileSize=*/true),
                 "could not open " + Path);
  }

  std::string Path;
};
}

// Create the default manifest file as a temporary file.
TemporaryFile createDefaultXml() {
  // Create a temporary file.
  TemporaryFile File("defaultxml", "manifest");

  // Open the temporary file for writing.
  std::error_code EC;
  llvm::raw_fd_ostream OS(File.Path, EC, sys::fs::F_Text);
  if (EC)
    fatal(EC, "failed to open " + File.Path);

  // Emit the XML. Note that we do *not* verify that the XML attributes are
  // syntactically correct. This is intentional for link.exe compatibility.
  OS << "<?xml version=\"1.0\" standalone=\"yes\"?>\n"
     << "<assembly xmlns=\"urn:schemas-microsoft-com:asm.v1\"\n"
     << "          manifestVersion=\"1.0\">\n";
  if (Config->ManifestUAC) {
    OS << "  <trustInfo>\n"
       << "    <security>\n"
       << "      <requestedPrivileges>\n"
       << "         <requestedExecutionLevel level=" << Config->ManifestLevel
       << " uiAccess=" << Config->ManifestUIAccess << "/>\n"
       << "      </requestedPrivileges>\n"
       << "    </security>\n"
       << "  </trustInfo>\n";
    if (!Config->ManifestDependency.empty()) {
      OS << "  <dependency>\n"
         << "    <dependentAssembly>\n"
         << "      <assemblyIdentity " << Config->ManifestDependency << " />\n"
         << "    </dependentAssembly>\n"
         << "  </dependency>\n";
    }
  }
  OS << "</assembly>\n";
  OS.close();
  return File;
}

static std::string readFile(StringRef Path) {
  std::unique_ptr<MemoryBuffer> MB =
      check(MemoryBuffer::getFile(Path), "could not open " + Path);
  return MB->getBuffer();
}

static std::string createManifestXml() {
  // Create the default manifest file.
  TemporaryFile File1 = createDefaultXml();
  if (Config->ManifestInput.empty())
    return readFile(File1.Path);

  // If manifest files are supplied by the user using /MANIFESTINPUT
  // option, we need to merge them with the default manifest.
  TemporaryFile File2("user", "manifest");

  Executor E("mt.exe");
  E.add("/manifest");
  E.add(File1.Path);
  for (StringRef Filename : Config->ManifestInput) {
    E.add("/manifest");
    E.add(Filename);
  }
  E.add("/nologo");
  E.add("/out:" + StringRef(File2.Path));
  E.run();
  return readFile(File2.Path);
}

// Create a resource file containing a manifest XML.
std::unique_ptr<MemoryBuffer> createManifestRes() {
  // Create a temporary file for the resource script file.
  TemporaryFile RCFile("manifest", "rc");

  // Open the temporary file for writing.
  std::error_code EC;
  llvm::raw_fd_ostream Out(RCFile.Path, EC, sys::fs::F_Text);
  if (EC)
    fatal(EC, "failed to open " + RCFile.Path);

  // Write resource script to the RC file.
  Out << "#define LANG_ENGLISH 9\n"
      << "#define SUBLANG_DEFAULT 1\n"
      << "#define APP_MANIFEST " << Config->ManifestID << "\n"
      << "#define RT_MANIFEST 24\n"
      << "LANGUAGE LANG_ENGLISH, SUBLANG_DEFAULT\n"
      << "APP_MANIFEST RT_MANIFEST {\n";
  quoteAndPrint(Out, createManifestXml());
  Out << "}\n";
  Out.close();

  // Create output resource file.
  TemporaryFile ResFile("output-resource", "res");

  Executor E("rc.exe");
  E.add("/fo");
  E.add(ResFile.Path);
  E.add("/nologo");
  E.add(RCFile.Path);
  E.run();
  return ResFile.getMemoryBuffer();
}

void createSideBySideManifest() {
  std::string Path = Config->ManifestFile;
  if (Path == "")
    Path = Config->OutputFile + ".manifest";
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::F_Text);
  if (EC)
    fatal(EC, "failed to create manifest");
  Out << createManifestXml();
}

// Parse a string in the form of
// "<name>[=<internalname>][,@ordinal[,NONAME]][,DATA][,PRIVATE]"
// or "<name>=<dllname>.<name>".
// Used for parsing /export arguments.
Export parseExport(StringRef Arg) {
  Export E;
  StringRef Rest;
  std::tie(E.Name, Rest) = Arg.split(",");
  if (E.Name.empty())
    goto err;

  if (E.Name.find('=') != StringRef::npos) {
    StringRef X, Y;
    std::tie(X, Y) = E.Name.split("=");

    // If "<name>=<dllname>.<name>".
    if (Y.find(".") != StringRef::npos) {
      E.Name = X;
      E.ForwardTo = Y;
      return E;
    }

    E.ExtName = X;
    E.Name = Y;
    if (E.Name.empty())
      goto err;
  }

  // If "<name>=<internalname>[,@ordinal[,NONAME]][,DATA][,PRIVATE]"
  while (!Rest.empty()) {
    StringRef Tok;
    std::tie(Tok, Rest) = Rest.split(",");
    if (Tok.equals_lower("noname")) {
      if (E.Ordinal == 0)
        goto err;
      E.Noname = true;
      continue;
    }
    if (Tok.equals_lower("data")) {
      E.Data = true;
      continue;
    }
    if (Tok.equals_lower("private")) {
      E.Private = true;
      continue;
    }
    if (Tok.startswith("@")) {
      int32_t Ord;
      if (Tok.substr(1).getAsInteger(0, Ord))
        goto err;
      if (Ord <= 0 || 65535 < Ord)
        goto err;
      E.Ordinal = Ord;
      continue;
    }
    goto err;
  }
  return E;

err:
  fatal("invalid /export: " + Arg);
}

static StringRef undecorate(StringRef Sym) {
  if (Config->Machine != I386)
    return Sym;
  return Sym.startswith("_") ? Sym.substr(1) : Sym;
}

// Performs error checking on all /export arguments.
// It also sets ordinals.
void fixupExports() {
  // Symbol ordinals must be unique.
  std::set<uint16_t> Ords;
  for (Export &E : Config->Exports) {
    if (E.Ordinal == 0)
      continue;
    if (!Ords.insert(E.Ordinal).second)
      fatal("duplicate export ordinal: " + E.Name);
  }

  for (Export &E : Config->Exports) {
    if (!E.ForwardTo.empty()) {
      E.SymbolName = E.Name;
    } else if (Undefined *U = cast_or_null<Undefined>(E.Sym->WeakAlias)) {
      E.SymbolName = U->getName();
    } else {
      E.SymbolName = E.Sym->getName();
    }
  }

  for (Export &E : Config->Exports) {
    if (!E.ForwardTo.empty()) {
      E.ExportName = undecorate(E.Name);
    } else {
      E.ExportName = undecorate(E.ExtName.empty() ? E.Name : E.ExtName);
    }
  }

  // Uniquefy by name.
  std::map<StringRef, Export *> Map;
  std::vector<Export> V;
  for (Export &E : Config->Exports) {
    auto Pair = Map.insert(std::make_pair(E.ExportName, &E));
    bool Inserted = Pair.second;
    if (Inserted) {
      V.push_back(E);
      continue;
    }
    Export *Existing = Pair.first->second;
    if (E == *Existing || E.Name != Existing->Name)
      continue;
    llvm::errs() << "warning: duplicate /export option: " << E.Name << "\n";
  }
  Config->Exports = std::move(V);

  // Sort by name.
  std::sort(Config->Exports.begin(), Config->Exports.end(),
            [](const Export &A, const Export &B) {
              return A.ExportName < B.ExportName;
            });
}

void assignExportOrdinals() {
  // Assign unique ordinals if default (= 0).
  uint16_t Max = 0;
  for (Export &E : Config->Exports)
    Max = std::max(Max, E.Ordinal);
  for (Export &E : Config->Exports)
    if (E.Ordinal == 0)
      E.Ordinal = ++Max;
}

// Parses a string in the form of "key=value" and check
// if value matches previous values for the same key.
void checkFailIfMismatch(StringRef Arg) {
  StringRef K, V;
  std::tie(K, V) = Arg.split('=');
  if (K.empty() || V.empty())
    fatal("/failifmismatch: invalid argument: " + Arg);
  StringRef Existing = Config->MustMatch[K];
  if (!Existing.empty() && V != Existing)
    fatal("/failifmismatch: mismatch detected: " + Existing + " and " + V +
          " for key " + K);
  Config->MustMatch[K] = V;
}

// Convert Windows resource files (.res files) to a .obj file
// using cvtres.exe.
std::unique_ptr<MemoryBuffer>
convertResToCOFF(const std::vector<MemoryBufferRef> &MBs) {
  // Create an output file path.
  TemporaryFile File("resource-file", "obj");

  // Execute cvtres.exe.
  Executor E("cvtres.exe");
  E.add("/machine:" + machineToStr(Config->Machine));
  E.add("/readonly");
  E.add("/nologo");
  E.add("/out:" + Twine(File.Path));
  for (MemoryBufferRef MB : MBs)
    E.add(MB.getBufferIdentifier());
  E.run();
  return File.getMemoryBuffer();
}

// Create OptTable

// Create prefix string literals used in Options.td
#define PREFIX(NAME, VALUE) const char *const NAME[] = VALUE;
#include "Options.inc"
#undef PREFIX

// Create table mapping all options defined in Options.td
static const llvm::opt::OptTable::Info infoTable[] = {
#define OPTION(X1, X2, ID, KIND, GROUP, ALIAS, X6, X7, X8, X9, X10)    \
  {                                                                    \
    X1, X2, X9, X10, OPT_##ID, llvm::opt::Option::KIND##Class, X8, X7, \
    OPT_##GROUP, OPT_##ALIAS, X6                                       \
  },
#include "Options.inc"
#undef OPTION
};

class COFFOptTable : public llvm::opt::OptTable {
public:
  COFFOptTable() : OptTable(infoTable, true) {}
};

// Parses a given list of options.
llvm::opt::InputArgList ArgParser::parse(ArrayRef<const char *> ArgsArr) {
  // First, replace respnose files (@<file>-style options).
  std::vector<const char *> Argv = replaceResponseFiles(ArgsArr);

  // Make InputArgList from string vectors.
  COFFOptTable Table;
  unsigned MissingIndex;
  unsigned MissingCount;
  llvm::opt::InputArgList Args =
      Table.ParseArgs(Argv, MissingIndex, MissingCount);

  // Print the real command line if response files are expanded.
  if (Args.hasArg(OPT_verbose) && ArgsArr.size() != Argv.size()) {
    llvm::outs() << "Command line:";
    for (const char *S : Argv)
      llvm::outs() << " " << S;
    llvm::outs() << "\n";
  }

  if (MissingCount)
    fatal("missing arg value for \"" + Twine(Args.getArgString(MissingIndex)) +
          "\", expected " + Twine(MissingCount) +
          (MissingCount == 1 ? " argument." : " arguments."));
  for (auto *Arg : Args.filtered(OPT_UNKNOWN))
    llvm::errs() << "ignoring unknown argument: " << Arg->getSpelling() << "\n";
  return Args;
}

llvm::opt::InputArgList ArgParser::parseLINK(ArrayRef<const char *> Args) {
  // Concatenate LINK env and given arguments and parse them.
  Optional<std::string> Env = Process::GetEnv("LINK");
  if (!Env)
    return parse(Args);
  std::vector<const char *> V = tokenize(*Env);
  V.insert(V.end(), Args.begin(), Args.end());
  return parse(V);
}

std::vector<const char *> ArgParser::tokenize(StringRef S) {
  SmallVector<const char *, 16> Tokens;
  StringSaver Saver(AllocAux);
  llvm::cl::TokenizeWindowsCommandLine(S, Saver, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

// Creates a new command line by replacing options starting with '@'
// character. '@<filename>' is replaced by the file's contents.
std::vector<const char *>
ArgParser::replaceResponseFiles(std::vector<const char *> Argv) {
  SmallVector<const char *, 256> Tokens(Argv.data(), Argv.data() + Argv.size());
  StringSaver Saver(AllocAux);
  ExpandResponseFiles(Saver, TokenizeWindowsCommandLine, Tokens);
  return std::vector<const char *>(Tokens.begin(), Tokens.end());
}

void printHelp(const char *Argv0) {
  COFFOptTable Table;
  Table.PrintHelp(llvm::outs(), Argv0, "LLVM Linker", false);
}

} // namespace coff
} // namespace lld
