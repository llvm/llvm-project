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
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/COFF.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
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
    error(ExeOrErr, Twine("unable to find ") + Prog + " in PATH: ");
    const char *Exe = Saver.save(*ExeOrErr);
    Args.insert(Args.begin(), Exe);
    Args.push_back(nullptr);
    if (llvm::sys::ExecuteAndWait(Args[0], Args.data()) != 0) {
      for (const char *S : Args)
        if (S)
          llvm::errs() << S << " ";
      error("failed");
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
  error(Twine("unknown /machine argument: ") + S);
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
    error(Twine("invalid number: ") + S1);
  if (Size && !S2.empty() && S2.getAsInteger(0, *Size))
    error(Twine("invalid number: ") + S2);
}

// Parses a string in the form of "<integer>[.<integer>]".
// If second number is not present, Minor is set to 0.
void parseVersion(StringRef Arg, uint32_t *Major, uint32_t *Minor) {
  StringRef S1, S2;
  std::tie(S1, S2) = Arg.split('.');
  if (S1.getAsInteger(0, *Major))
    error(Twine("invalid number: ") + S1);
  *Minor = 0;
  if (!S2.empty() && S2.getAsInteger(0, *Minor))
    error(Twine("invalid number: ") + S2);
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
    error(Twine("unknown subsystem: ") + SysStr);
  if (!Ver.empty())
    parseVersion(Ver, Major, Minor);
}

// Parse a string of the form of "<from>=<to>".
// Results are directly written to Config.
void parseAlternateName(StringRef S) {
  StringRef From, To;
  std::tie(From, To) = S.split('=');
  if (From.empty() || To.empty())
    error(Twine("/alternatename: invalid argument: ") + S);
  auto It = Config->AlternateNames.find(From);
  if (It != Config->AlternateNames.end() && It->second != To)
    error(Twine("/alternatename: conflicts: ") + S);
  Config->AlternateNames.insert(It, std::make_pair(From, To));
}

// Parse a string of the form of "<from>=<to>".
// Results are directly written to Config.
void parseMerge(StringRef S) {
  StringRef From, To;
  std::tie(From, To) = S.split('=');
  if (From.empty() || To.empty())
    error(Twine("/merge: invalid argument: ") + S);
  auto Pair = Config->Merge.insert(std::make_pair(From, To));
  bool Inserted = Pair.second;
  if (!Inserted) {
    StringRef Existing = Pair.first->second;
    if (Existing != To)
      llvm::errs() << "warning: " << S << ": already merged into "
                   << Existing << "\n";
  }
}

// Parses a string in the form of "EMBED[,=<integer>]|NO".
// Results are directly written to Config.
void parseManifest(StringRef Arg) {
  if (Arg.equals_lower("no")) {
    Config->Manifest = Configuration::No;
    return;
  }
  if (!Arg.startswith_lower("embed"))
    error(Twine("Invalid option ") + Arg);
  Config->Manifest = Configuration::Embed;
  Arg = Arg.substr(strlen("embed"));
  if (Arg.empty())
    return;
  if (!Arg.startswith_lower(",id="))
    error(Twine("Invalid option ") + Arg);
  Arg = Arg.substr(strlen(",id="));
  if (Arg.getAsInteger(0, Config->ManifestID))
    error(Twine("Invalid option ") + Arg);
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
    error(Twine("Invalid option ") + Arg);
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

// Create a manifest file contents.
static std::string createManifestXml() {
  std::string S;
  llvm::raw_string_ostream OS(S);
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
  OS.flush();
  return S;
}

// Create a resource file containing a manifest XML.
std::unique_ptr<MemoryBuffer> createManifestRes() {
  // Create a temporary file for the resource script file.
  SmallString<128> RCPath;
  std::error_code EC = sys::fs::createTemporaryFile("tmp", "rc", RCPath);
  error(EC, "cannot create a temporary file");
  FileRemover RCRemover(RCPath);

  // Open the temporary file for writing.
  llvm::raw_fd_ostream Out(RCPath, EC, sys::fs::F_Text);
  error(EC, Twine("failed to open ") + RCPath);

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
  SmallString<128> ResPath;
  EC = sys::fs::createTemporaryFile("tmp", "res", ResPath);
  error(EC, "cannot create a temporary file");

  Executor E("rc.exe");
  E.add("/fo");
  E.add(ResPath.str());
  E.add("/nologo");
  E.add(RCPath.str());
  E.run();
  ErrorOr<std::unique_ptr<MemoryBuffer>> Ret = MemoryBuffer::getFile(ResPath);
  error(Ret, Twine("Could not open ") + ResPath);
  return std::move(*Ret);
}

void createSideBySideManifest() {
  std::string Path = Config->ManifestFile;
  if (Path == "")
    Path = (Twine(Config->OutputFile) + ".manifest").str();
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::F_Text);
  error(EC, "failed to create manifest");
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
  error(Twine("invalid /export: ") + Arg);
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
      error("duplicate export ordinal: " + E.Name);
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
    error(Twine("/failifmismatch: invalid argument: ") + Arg);
  StringRef Existing = Config->MustMatch[K];
  if (!Existing.empty() && V != Existing)
    error(Twine("/failifmismatch: mismatch detected: ") + Existing + " and " +
          V + " for key " + K);
  Config->MustMatch[K] = V;
}

// Convert Windows resource files (.res files) to a .obj file
// using cvtres.exe.
std::unique_ptr<MemoryBuffer>
convertResToCOFF(const std::vector<MemoryBufferRef> &MBs) {
  // Create an output file path.
  SmallString<128> Path;
  if (llvm::sys::fs::createTemporaryFile("resource", "obj", Path))
    error("Could not create temporary file");

  // Execute cvtres.exe.
  Executor E("cvtres.exe");
  E.add("/machine:" + machineToStr(Config->Machine));
  E.add("/readonly");
  E.add("/nologo");
  E.add("/out:" + Path);
  for (MemoryBufferRef MB : MBs)
    E.add(MB.getBufferIdentifier());
  E.run();
  ErrorOr<std::unique_ptr<MemoryBuffer>> Ret = MemoryBuffer::getFile(Path);
  error(Ret, Twine("Could not open ") + Path);
  return std::move(*Ret);
}

static std::string writeToTempFile(StringRef Contents) {
  SmallString<128> Path;
  int FD;
  if (llvm::sys::fs::createTemporaryFile("tmp", "def", FD, Path)) {
    llvm::errs() << "failed to create a temporary file\n";
    return "";
  }
  llvm::raw_fd_ostream OS(FD, /*shouldClose*/ true);
  OS << Contents;
  return Path.str();
}

void touchFile(StringRef Path) {
  int FD;
  std::error_code EC = sys::fs::openFileForWrite(Path, FD, sys::fs::F_Append);
  error(EC, "failed to create a file");
  sys::Process::SafelyCloseFileDescriptor(FD);
}

static std::string getImplibPath() {
  if (!Config->Implib.empty())
    return Config->Implib;
  SmallString<128> Out = StringRef(Config->OutputFile);
  sys::path::replace_extension(Out, ".lib");
  return Out.str();
}

static std::unique_ptr<MemoryBuffer> createEmptyImportLibrary() {
  std::string S = (Twine("LIBRARY \"") +
                   llvm::sys::path::filename(Config->OutputFile) + "\"\n")
                      .str();
  std::string Path1 = writeToTempFile(S);
  std::string Path2 = getImplibPath();
  llvm::FileRemover Remover1(Path1);
  llvm::FileRemover Remover2(Path2);

  Executor E("lib.exe");
  E.add("/nologo");
  E.add("/machine:" + machineToStr(Config->Machine));
  E.add(Twine("/def:") + Path1);
  E.add(Twine("/out:") + Path2);
  E.run();

  ErrorOr<std::unique_ptr<MemoryBuffer>> BufOrErr =
      MemoryBuffer::getFile(Path2, -1, false);
  error(BufOrErr, Twine("Failed to open ") + Path2);
  return MemoryBuffer::getMemBufferCopy((*BufOrErr)->getBuffer());
}

static std::vector<NewArchiveIterator>
readMembers(const object::Archive &Archive) {
  std::vector<NewArchiveIterator> V;
  for (const auto &ChildOrErr : Archive.children()) {
    error(ChildOrErr, "Archive::Child::getName failed");
    const object::Archive::Child C(*ChildOrErr);
    ErrorOr<StringRef> NameOrErr = C.getName();
    error(NameOrErr, "Archive::Child::getName failed");
    V.emplace_back(C, *NameOrErr);
  }
  return V;
}

// This class creates short import files which is described in
// PE/COFF spec 7. Import Library Format.
class ShortImportCreator {
public:
  ShortImportCreator(object::Archive *A, StringRef S) : Parent(A), DLLName(S) {}

  NewArchiveIterator create(StringRef Sym, uint16_t Ordinal,
                            ImportNameType NameType, bool isData) {
    size_t ImpSize = DLLName.size() + Sym.size() + 2; // +2 for NULs
    size_t Size = sizeof(object::ArchiveMemberHeader) +
                  sizeof(coff_import_header) + ImpSize;
    char *Buf = Alloc.Allocate<char>(Size);
    memset(Buf, 0, Size);
    char *P = Buf;

    // Write archive member header
    auto *Hdr = reinterpret_cast<object::ArchiveMemberHeader *>(P);
    P += sizeof(*Hdr);
    sprintf(Hdr->Name, "%-12s", "dummy");
    sprintf(Hdr->LastModified, "%-12d", 0);
    sprintf(Hdr->UID, "%-6d", 0);
    sprintf(Hdr->GID, "%-6d", 0);
    sprintf(Hdr->AccessMode, "%-8d", 0644);
    sprintf(Hdr->Size, "%-10d", int(sizeof(coff_import_header) + ImpSize));

    // Write short import library.
    auto *Imp = reinterpret_cast<coff_import_header *>(P);
    P += sizeof(*Imp);
    Imp->Sig2 = 0xFFFF;
    Imp->Machine = Config->Machine;
    Imp->SizeOfData = ImpSize;
    if (Ordinal > 0)
      Imp->OrdinalHint = Ordinal;
    Imp->TypeInfo = (isData ? IMPORT_DATA : IMPORT_CODE);
    Imp->TypeInfo |= NameType << 2;

    // Write symbol name and DLL name.
    memcpy(P, Sym.data(), Sym.size());
    P += Sym.size() + 1;
    memcpy(P, DLLName.data(), DLLName.size());

    std::error_code EC;
    object::Archive::Child C(Parent, Buf, &EC);
    assert(!EC && "We created an invalid buffer");
    return NewArchiveIterator(C, DLLName);
  }

private:
  BumpPtrAllocator Alloc;
  object::Archive *Parent;
  StringRef DLLName;
};

static ImportNameType getNameType(StringRef Sym, StringRef ExtName) {
  if (Sym != ExtName)
    return IMPORT_NAME_UNDECORATE;
  if (Config->Machine == I386 && Sym.startswith("_"))
    return IMPORT_NAME_NOPREFIX;
  return IMPORT_NAME;
}

static std::string replace(StringRef S, StringRef From, StringRef To) {
  size_t Pos = S.find(From);
  assert(Pos != StringRef::npos);
  return (Twine(S.substr(0, Pos)) + To + S.substr(Pos + From.size())).str();
}

// Creates an import library for a DLL. In this function, we first
// create an empty import library using lib.exe and then adds short
// import files to that file.
void writeImportLibrary() {
  std::unique_ptr<MemoryBuffer> Buf = createEmptyImportLibrary();
  std::error_code EC;
  object::Archive Archive(Buf->getMemBufferRef(), EC);
  error(EC, "Error reading an empty import file");
  std::vector<NewArchiveIterator> Members = readMembers(Archive);

  std::string DLLName = llvm::sys::path::filename(Config->OutputFile);
  ShortImportCreator ShortImport(&Archive, DLLName);
  for (Export &E : Config->Exports) {
    if (E.Private)
      continue;
    if (E.ExtName.empty()) {
      Members.push_back(ShortImport.create(
          E.SymbolName, E.Ordinal, getNameType(E.SymbolName, E.Name), E.Data));
    } else {
      Members.push_back(ShortImport.create(
          replace(E.SymbolName, E.Name, E.ExtName), E.Ordinal,
          getNameType(E.SymbolName, E.Name), E.Data));
    }
  }

  std::string Path = getImplibPath();
  std::pair<StringRef, std::error_code> Result =
      writeArchive(Path, Members, /*WriteSymtab*/ true, object::Archive::K_GNU,
                   /*Deterministic*/ true, /*Thin*/ false);
  error(Result.second, Twine("Failed to write ") + Path);
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
    error(Twine("missing arg value for \"") + Args.getArgString(MissingIndex) +
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
