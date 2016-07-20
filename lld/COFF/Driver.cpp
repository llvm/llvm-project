//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Config.h"
#include "Driver.h"
#include "Error.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Writer.h"
#include "lld/Driver/Driver.h"
#include "llvm/ADT/Optional.h"
#include "llvm/LibDriver/LibDriver.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/Option.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

using namespace llvm;
using namespace llvm::COFF;
using llvm::sys::Process;
using llvm::sys::fs::OpenFlags;
using llvm::sys::fs::file_magic;
using llvm::sys::fs::identify_magic;

namespace lld {
namespace coff {

Configuration *Config;
LinkerDriver *Driver;

bool link(llvm::ArrayRef<const char *> Args) {
  Configuration C;
  LinkerDriver D;
  Config = &C;
  Driver = &D;
  Driver->link(Args);
  return true;
}

// Drop directory components and replace extension with ".exe" or ".dll".
static std::string getOutputPath(StringRef Path) {
  auto P = Path.find_last_of("\\/");
  StringRef S = (P == StringRef::npos) ? Path : Path.substr(P + 1);
  const char* E = Config->DLL ? ".dll" : ".exe";
  return (S.substr(0, S.rfind('.')) + E).str();
}

// Opens a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
MemoryBufferRef LinkerDriver::openFile(StringRef Path) {
  std::unique_ptr<MemoryBuffer> MB =
      check(MemoryBuffer::getFile(Path), "could not open " + Path);
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take ownership
  return MBRef;
}

static std::unique_ptr<InputFile> createFile(MemoryBufferRef MB) {
  // File type is detected by contents, not by file extension.
  file_magic Magic = identify_magic(MB.getBuffer());
  if (Magic == file_magic::archive)
    return std::unique_ptr<InputFile>(new ArchiveFile(MB));
  if (Magic == file_magic::bitcode)
    return std::unique_ptr<InputFile>(new BitcodeFile(MB));
  if (Config->OutputFile == "")
    Config->OutputFile = getOutputPath(MB.getBufferIdentifier());
  return std::unique_ptr<InputFile>(new ObjectFile(MB));
}

static bool isDecorated(StringRef Sym) {
  return Sym.startswith("_") || Sym.startswith("@") || Sym.startswith("?");
}

// Parses .drectve section contents and returns a list of files
// specified by /defaultlib.
void LinkerDriver::parseDirectives(StringRef S) {
  llvm::opt::InputArgList Args = Parser.parse(S);

  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_alternatename:
      parseAlternateName(Arg->getValue());
      break;
    case OPT_defaultlib:
      if (Optional<StringRef> Path = findLib(Arg->getValue())) {
        MemoryBufferRef MB = openFile(*Path);
        Symtab.addFile(createFile(MB));
      }
      break;
    case OPT_export: {
      Export E = parseExport(Arg->getValue());
      E.Directives = true;
      Config->Exports.push_back(E);
      break;
    }
    case OPT_failifmismatch:
      checkFailIfMismatch(Arg->getValue());
      break;
    case OPT_incl:
      addUndefined(Arg->getValue());
      break;
    case OPT_merge:
      parseMerge(Arg->getValue());
      break;
    case OPT_nodefaultlib:
      Config->NoDefaultLibs.insert(doFindLib(Arg->getValue()));
      break;
    case OPT_section:
      parseSection(Arg->getValue());
      break;
    case OPT_editandcontinue:
    case OPT_fastfail:
    case OPT_guardsym:
    case OPT_throwingnew:
      break;
    default:
      fatal(Arg->getSpelling() + " is not allowed in .drectve");
    }
  }
}

// Find file from search paths. You can omit ".obj", this function takes
// care of that. Note that the returned path is not guaranteed to exist.
StringRef LinkerDriver::doFindFile(StringRef Filename) {
  bool hasPathSep = (Filename.find_first_of("/\\") != StringRef::npos);
  if (hasPathSep)
    return Filename;
  bool hasExt = (Filename.find('.') != StringRef::npos);
  for (StringRef Dir : SearchPaths) {
    SmallString<128> Path = Dir;
    llvm::sys::path::append(Path, Filename);
    if (llvm::sys::fs::exists(Path.str()))
      return Alloc.save(Path.str());
    if (!hasExt) {
      Path.append(".obj");
      if (llvm::sys::fs::exists(Path.str()))
        return Alloc.save(Path.str());
    }
  }
  return Filename;
}

// Resolves a file path. This never returns the same path
// (in that case, it returns None).
Optional<StringRef> LinkerDriver::findFile(StringRef Filename) {
  StringRef Path = doFindFile(Filename);
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

// Find library file from search path.
StringRef LinkerDriver::doFindLib(StringRef Filename) {
  // Add ".lib" to Filename if that has no file extension.
  bool hasExt = (Filename.find('.') != StringRef::npos);
  if (!hasExt)
    Filename = Alloc.save(Filename + ".lib");
  return doFindFile(Filename);
}

// Resolves a library path. /nodefaultlib options are taken into
// consideration. This never returns the same path (in that case,
// it returns None).
Optional<StringRef> LinkerDriver::findLib(StringRef Filename) {
  if (Config->NoDefaultLibAll)
    return None;
  StringRef Path = doFindLib(Filename);
  if (Config->NoDefaultLibs.count(Path))
    return None;
  bool Seen = !VisitedFiles.insert(Path.lower()).second;
  if (Seen)
    return None;
  return Path;
}

// Parses LIB environment which contains a list of search paths.
void LinkerDriver::addLibSearchPaths() {
  Optional<std::string> EnvOpt = Process::GetEnv("LIB");
  if (!EnvOpt.hasValue())
    return;
  StringRef Env = Alloc.save(*EnvOpt);
  while (!Env.empty()) {
    StringRef Path;
    std::tie(Path, Env) = Env.split(';');
    SearchPaths.push_back(Path);
  }
}

Undefined *LinkerDriver::addUndefined(StringRef Name) {
  Undefined *U = Symtab.addUndefined(Name);
  Config->GCRoot.insert(U);
  return U;
}

// Symbol names are mangled by appending "_" prefix on x86.
StringRef LinkerDriver::mangle(StringRef Sym) {
  assert(Config->Machine != IMAGE_FILE_MACHINE_UNKNOWN);
  if (Config->Machine == I386)
    return Alloc.save("_" + Sym);
  return Sym;
}

// Windows specific -- find default entry point name.
StringRef LinkerDriver::findDefaultEntry() {
  // User-defined main functions and their corresponding entry points.
  static const char *Entries[][2] = {
      {"main", "mainCRTStartup"},
      {"wmain", "wmainCRTStartup"},
      {"WinMain", "WinMainCRTStartup"},
      {"wWinMain", "wWinMainCRTStartup"},
  };
  for (auto E : Entries) {
    StringRef Entry = Symtab.findMangle(mangle(E[0]));
    if (!Entry.empty() && !isa<Undefined>(Symtab.find(Entry)->Body))
      return mangle(E[1]);
  }
  return "";
}

WindowsSubsystem LinkerDriver::inferSubsystem() {
  if (Config->DLL)
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  if (Symtab.findUnderscore("main") || Symtab.findUnderscore("wmain"))
    return IMAGE_SUBSYSTEM_WINDOWS_CUI;
  if (Symtab.findUnderscore("WinMain") || Symtab.findUnderscore("wWinMain"))
    return IMAGE_SUBSYSTEM_WINDOWS_GUI;
  return IMAGE_SUBSYSTEM_UNKNOWN;
}

static uint64_t getDefaultImageBase() {
  if (Config->is64())
    return Config->DLL ? 0x180000000 : 0x140000000;
  return Config->DLL ? 0x10000000 : 0x400000;
}

void LinkerDriver::link(llvm::ArrayRef<const char *> ArgsArr) {
  // If the first command line argument is "/lib", link.exe acts like lib.exe.
  // We call our own implementation of lib.exe that understands bitcode files.
  if (ArgsArr.size() > 1 && StringRef(ArgsArr[1]).equals_lower("/lib")) {
    if (llvm::libDriverMain(ArgsArr.slice(1)) != 0)
      fatal("lib failed");
    return;
  }

  // Needed for LTO.
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllDisassemblers();

  // Parse command line options.
  llvm::opt::InputArgList Args = Parser.parseLINK(ArgsArr.slice(1));

  // Handle /help
  if (Args.hasArg(OPT_help)) {
    printHelp(ArgsArr[0]);
    return;
  }

  if (Args.filtered_begin(OPT_INPUT) == Args.filtered_end())
    fatal("no input files");

  // Construct search path list.
  SearchPaths.push_back("");
  for (auto *Arg : Args.filtered(OPT_libpath))
    SearchPaths.push_back(Arg->getValue());
  addLibSearchPaths();

  // Handle /out
  if (auto *Arg = Args.getLastArg(OPT_out))
    Config->OutputFile = Arg->getValue();

  // Handle /verbose
  if (Args.hasArg(OPT_verbose))
    Config->Verbose = true;

  // Handle /force or /force:unresolved
  if (Args.hasArg(OPT_force) || Args.hasArg(OPT_force_unresolved))
    Config->Force = true;

  // Handle /debug
  if (Args.hasArg(OPT_debug))
    Config->Debug = true;

  // Handle /noentry
  if (Args.hasArg(OPT_noentry)) {
    if (!Args.hasArg(OPT_dll))
      fatal("/noentry must be specified with /dll");
    Config->NoEntry = true;
  }

  // Handle /dll
  if (Args.hasArg(OPT_dll)) {
    Config->DLL = true;
    Config->ManifestID = 2;
  }

  // Handle /fixed
  if (Args.hasArg(OPT_fixed)) {
    if (Args.hasArg(OPT_dynamicbase))
      fatal("/fixed must not be specified with /dynamicbase");
    Config->Relocatable = false;
    Config->DynamicBase = false;
  }

  // Handle /machine
  if (auto *Arg = Args.getLastArg(OPT_machine))
    Config->Machine = getMachineType(Arg->getValue());

  // Handle /nodefaultlib:<filename>
  for (auto *Arg : Args.filtered(OPT_nodefaultlib))
    Config->NoDefaultLibs.insert(doFindLib(Arg->getValue()));

  // Handle /nodefaultlib
  if (Args.hasArg(OPT_nodefaultlib_all))
    Config->NoDefaultLibAll = true;

  // Handle /base
  if (auto *Arg = Args.getLastArg(OPT_base))
    parseNumbers(Arg->getValue(), &Config->ImageBase);

  // Handle /stack
  if (auto *Arg = Args.getLastArg(OPT_stack))
    parseNumbers(Arg->getValue(), &Config->StackReserve, &Config->StackCommit);

  // Handle /heap
  if (auto *Arg = Args.getLastArg(OPT_heap))
    parseNumbers(Arg->getValue(), &Config->HeapReserve, &Config->HeapCommit);

  // Handle /version
  if (auto *Arg = Args.getLastArg(OPT_version))
    parseVersion(Arg->getValue(), &Config->MajorImageVersion,
                 &Config->MinorImageVersion);

  // Handle /subsystem
  if (auto *Arg = Args.getLastArg(OPT_subsystem))
    parseSubsystem(Arg->getValue(), &Config->Subsystem, &Config->MajorOSVersion,
                   &Config->MinorOSVersion);

  // Handle /alternatename
  for (auto *Arg : Args.filtered(OPT_alternatename))
    parseAlternateName(Arg->getValue());

  // Handle /include
  for (auto *Arg : Args.filtered(OPT_incl))
    addUndefined(Arg->getValue());

  // Handle /implib
  if (auto *Arg = Args.getLastArg(OPT_implib))
    Config->Implib = Arg->getValue();

  // Handle /opt
  for (auto *Arg : Args.filtered(OPT_opt)) {
    std::string Str = StringRef(Arg->getValue()).lower();
    SmallVector<StringRef, 1> Vec;
    StringRef(Str).split(Vec, ',');
    for (StringRef S : Vec) {
      if (S == "noref") {
        Config->DoGC = false;
        Config->DoICF = false;
        continue;
      }
      if (S == "icf" || StringRef(S).startswith("icf=")) {
        Config->DoICF = true;
        continue;
      }
      if (S == "noicf") {
        Config->DoICF = false;
        continue;
      }
      if (StringRef(S).startswith("lldlto=")) {
        StringRef OptLevel = StringRef(S).substr(7);
        if (OptLevel.getAsInteger(10, Config->LTOOptLevel) ||
            Config->LTOOptLevel > 3)
          fatal("/opt:lldlto: invalid optimization level: " + OptLevel);
        continue;
      }
      if (StringRef(S).startswith("lldltojobs=")) {
        StringRef Jobs = StringRef(S).substr(11);
        if (Jobs.getAsInteger(10, Config->LTOJobs) || Config->LTOJobs == 0)
          fatal("/opt:lldltojobs: invalid job count: " + Jobs);
        continue;
      }
      if (S != "ref" && S != "lbr" && S != "nolbr")
        fatal("/opt: unknown option: " + S);
    }
  }

  // Handle /failifmismatch
  for (auto *Arg : Args.filtered(OPT_failifmismatch))
    checkFailIfMismatch(Arg->getValue());

  // Handle /merge
  for (auto *Arg : Args.filtered(OPT_merge))
    parseMerge(Arg->getValue());

  // Handle /section
  for (auto *Arg : Args.filtered(OPT_section))
    parseSection(Arg->getValue());

  // Handle /manifest
  if (auto *Arg = Args.getLastArg(OPT_manifest_colon))
    parseManifest(Arg->getValue());

  // Handle /manifestuac
  if (auto *Arg = Args.getLastArg(OPT_manifestuac))
    parseManifestUAC(Arg->getValue());

  // Handle /manifestdependency
  if (auto *Arg = Args.getLastArg(OPT_manifestdependency))
    Config->ManifestDependency = Arg->getValue();

  // Handle /manifestfile
  if (auto *Arg = Args.getLastArg(OPT_manifestfile))
    Config->ManifestFile = Arg->getValue();

  // Handle /manifestinput
  for (auto *Arg : Args.filtered(OPT_manifestinput))
    Config->ManifestInput.push_back(Arg->getValue());

  // Handle miscellaneous boolean flags.
  if (Args.hasArg(OPT_allowbind_no))
    Config->AllowBind = false;
  if (Args.hasArg(OPT_allowisolation_no))
    Config->AllowIsolation = false;
  if (Args.hasArg(OPT_dynamicbase_no))
    Config->DynamicBase = false;
  if (Args.hasArg(OPT_nxcompat_no))
    Config->NxCompat = false;
  if (Args.hasArg(OPT_tsaware_no))
    Config->TerminalServerAware = false;
  if (Args.hasArg(OPT_nosymtab))
    Config->WriteSymtab = false;

  // Create a list of input files. Files can be given as arguments
  // for /defaultlib option.
  std::vector<StringRef> Paths;
  std::vector<MemoryBufferRef> MBs;
  for (auto *Arg : Args.filtered(OPT_INPUT))
    if (Optional<StringRef> Path = findFile(Arg->getValue()))
      Paths.push_back(*Path);
  for (auto *Arg : Args.filtered(OPT_defaultlib))
    if (Optional<StringRef> Path = findLib(Arg->getValue()))
      Paths.push_back(*Path);
  for (StringRef Path : Paths)
    MBs.push_back(openFile(Path));

  // Windows specific -- Create a resource file containing a manifest file.
  if (Config->Manifest == Configuration::Embed) {
    std::unique_ptr<MemoryBuffer> MB = createManifestRes();
    MBs.push_back(MB->getMemBufferRef());
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Windows specific -- Input files can be Windows resource files (.res files).
  // We invoke cvtres.exe to convert resource files to a regular COFF file
  // then link the result file normally.
  std::vector<MemoryBufferRef> Resources;
  auto NotResource = [](MemoryBufferRef MB) {
    return identify_magic(MB.getBuffer()) != file_magic::windows_resource;
  };
  auto It = std::stable_partition(MBs.begin(), MBs.end(), NotResource);
  if (It != MBs.end()) {
    Resources.insert(Resources.end(), It, MBs.end());
    MBs.erase(It, MBs.end());
  }

  // Read all input files given via the command line. Note that step()
  // doesn't read files that are specified by directive sections.
  for (MemoryBufferRef MB : MBs)
    Symtab.addFile(createFile(MB));
  Symtab.step();

  // Determine machine type and check if all object files are
  // for the same CPU type. Note that this needs to be done before
  // any call to mangle().
  for (std::unique_ptr<InputFile> &File : Symtab.getFiles()) {
    MachineTypes MT = File->getMachineType();
    if (MT == IMAGE_FILE_MACHINE_UNKNOWN)
      continue;
    if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
      Config->Machine = MT;
      continue;
    }
    if (Config->Machine != MT)
      fatal(File->getShortName() + ": machine type " + machineToStr(MT) +
            " conflicts with " + machineToStr(Config->Machine));
  }
  if (Config->Machine == IMAGE_FILE_MACHINE_UNKNOWN) {
    llvm::errs() << "warning: /machine is not specified. x64 is assumed.\n";
    Config->Machine = AMD64;
  }

  // Windows specific -- Convert Windows resource files to a COFF file.
  if (!Resources.empty()) {
    std::unique_ptr<MemoryBuffer> MB = convertResToCOFF(Resources);
    Symtab.addFile(createFile(MB->getMemBufferRef()));
    OwningMBs.push_back(std::move(MB)); // take ownership
  }

  // Handle /largeaddressaware
  if (Config->is64() || Args.hasArg(OPT_largeaddressaware))
    Config->LargeAddressAware = true;

  // Handle /highentropyva
  if (Config->is64() && !Args.hasArg(OPT_highentropyva_no))
    Config->HighEntropyVA = true;

  // Handle /entry and /dll
  if (auto *Arg = Args.getLastArg(OPT_entry)) {
    Config->Entry = addUndefined(mangle(Arg->getValue()));
  } else if (Args.hasArg(OPT_dll) && !Config->NoEntry) {
    StringRef S = (Config->Machine == I386) ? "__DllMainCRTStartup@12"
                                            : "_DllMainCRTStartup";
    Config->Entry = addUndefined(S);
  } else if (!Config->NoEntry) {
    // Windows specific -- If entry point name is not given, we need to
    // infer that from user-defined entry name.
    StringRef S = findDefaultEntry();
    if (S.empty())
      fatal("entry point must be defined");
    Config->Entry = addUndefined(S);
    if (Config->Verbose)
      llvm::outs() << "Entry name inferred: " << S << "\n";
  }

  // Handle /export
  for (auto *Arg : Args.filtered(OPT_export)) {
    Export E = parseExport(Arg->getValue());
    if (Config->Machine == I386) {
      if (!isDecorated(E.Name))
        E.Name = Alloc.save("_" + E.Name);
      if (!E.ExtName.empty() && !isDecorated(E.ExtName))
        E.ExtName = Alloc.save("_" + E.ExtName);
    }
    Config->Exports.push_back(E);
  }

  // Handle /def
  if (auto *Arg = Args.getLastArg(OPT_deffile)) {
    MemoryBufferRef MB = openFile(Arg->getValue());
    // parseModuleDefs mutates Config object.
    parseModuleDefs(MB, &Alloc);
  }

  // Handle /delayload
  for (auto *Arg : Args.filtered(OPT_delayload)) {
    Config->DelayLoads.insert(StringRef(Arg->getValue()).lower());
    if (Config->Machine == I386) {
      Config->DelayLoadHelper = addUndefined("___delayLoadHelper2@8");
    } else {
      Config->DelayLoadHelper = addUndefined("__delayLoadHelper2");
    }
  }

  // Set default image base if /base is not given.
  if (Config->ImageBase == uint64_t(-1))
    Config->ImageBase = getDefaultImageBase();

  Symtab.addRelative(mangle("__ImageBase"), 0);
  if (Config->Machine == I386) {
    Config->SEHTable = Symtab.addRelative("___safe_se_handler_table", 0);
    Config->SEHCount = Symtab.addAbsolute("___safe_se_handler_count", 0);
  }

  // We do not support /guard:cf (control flow protection) yet.
  // Define CFG symbols anyway so that we can link MSVC 2015 CRT.
  Symtab.addAbsolute(mangle("__guard_fids_table"), 0);
  Symtab.addAbsolute(mangle("__guard_fids_count"), 0);
  Symtab.addAbsolute(mangle("__guard_flags"), 0x100);

  // Read as much files as we can from directives sections.
  Symtab.run();

  // Resolve auxiliary symbols until we get a convergence.
  // (Trying to resolve a symbol may trigger a Lazy symbol to load a new file.
  // A new file may contain a directive section to add new command line options.
  // That's why we have to repeat until converge.)
  for (;;) {
    // Windows specific -- if entry point is not found,
    // search for its mangled names.
    if (Config->Entry)
      Symtab.mangleMaybe(Config->Entry);

    // Windows specific -- Make sure we resolve all dllexported symbols.
    for (Export &E : Config->Exports) {
      if (!E.ForwardTo.empty())
        continue;
      E.Sym = addUndefined(E.Name);
      if (!E.Directives)
        Symtab.mangleMaybe(E.Sym);
    }

    // Add weak aliases. Weak aliases is a mechanism to give remaining
    // undefined symbols final chance to be resolved successfully.
    for (auto Pair : Config->AlternateNames) {
      StringRef From = Pair.first;
      StringRef To = Pair.second;
      Symbol *Sym = Symtab.find(From);
      if (!Sym)
        continue;
      if (auto *U = dyn_cast<Undefined>(Sym->Body))
        if (!U->WeakAlias)
          U->WeakAlias = Symtab.addUndefined(To);
    }

    // Windows specific -- if __load_config_used can be resolved, resolve it.
    if (Symtab.findUnderscore("_load_config_used"))
      addUndefined(mangle("_load_config_used"));

    if (Symtab.queueEmpty())
      break;
    Symtab.run();
  }

  // Do LTO by compiling bitcode input files to a set of native COFF files then
  // link those files.
  Symtab.addCombinedLTOObjects();

  // Make sure we have resolved all symbols.
  Symtab.reportRemainingUndefines(/*Resolve=*/true);

  // Windows specific -- if no /subsystem is given, we need to infer
  // that from entry point name.
  if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN) {
    Config->Subsystem = inferSubsystem();
    if (Config->Subsystem == IMAGE_SUBSYSTEM_UNKNOWN)
      fatal("subsystem must be defined");
  }

  // Handle /safeseh.
  if (Args.hasArg(OPT_safeseh))
    for (ObjectFile *File : Symtab.ObjectFiles)
      if (!File->SEHCompat)
        fatal("/safeseh: " + File->getName() + " is not compatible with SEH");

  // Windows specific -- when we are creating a .dll file, we also
  // need to create a .lib file.
  if (!Config->Exports.empty() || Config->DLL) {
    fixupExports();
    writeImportLibrary();
    assignExportOrdinals();
  }

  // Windows specific -- Create a side-by-side manifest file.
  if (Config->Manifest == Configuration::SideBySide)
    createSideBySideManifest();

  // Create a dummy PDB file to satisfy build sytem rules.
  if (auto *Arg = Args.getLastArg(OPT_pdb))
    createPDB(Arg->getValue());

  // Identify unreferenced COMDAT sections.
  if (Config->DoGC)
    markLive(Symtab.getChunks());

  // Identify identical COMDAT sections to merge them.
  if (Config->DoICF)
    doICF(Symtab.getChunks());

  // Write the result.
  writeResult(&Symtab);

  // Create a symbol map file containing symbol VAs and their names
  // to help debugging.
  if (auto *Arg = Args.getLastArg(OPT_lldmap)) {
    std::error_code EC;
    llvm::raw_fd_ostream Out(Arg->getValue(), EC, OpenFlags::F_Text);
    if (EC)
      fatal(EC, "could not create the symbol map");
    Symtab.printMap(Out);
  }
  // Call exit to avoid calling destructors.
  exit(0);
}

} // namespace coff
} // namespace lld
