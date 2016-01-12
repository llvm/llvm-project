//===- Driver.cpp ---------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Driver.h"
#include "Config.h"
#include "Error.h"
#include "InputFiles.h"
#include "SymbolTable.h"
#include "Target.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf2;

Configuration *elf2::Config;
LinkerDriver *elf2::Driver;

void elf2::link(ArrayRef<const char *> Args) {
  Configuration C;
  LinkerDriver D;
  Config = &C;
  Driver = &D;
  Driver->main(Args.slice(1));
}

static std::pair<ELFKind, uint16_t> parseEmulation(StringRef S) {
  if (S == "elf32btsmip")
    return {ELF32BEKind, EM_MIPS};
  if (S == "elf32ltsmip")
    return {ELF32LEKind, EM_MIPS};
  if (S == "elf32ppc" || S == "elf32ppc_fbsd")
    return {ELF32BEKind, EM_PPC};
  if (S == "elf64ppc" || S == "elf64ppc_fbsd")
    return {ELF64BEKind, EM_PPC64};
  if (S == "elf_i386")
    return {ELF32LEKind, EM_386};
  if (S == "elf_x86_64")
    return {ELF64LEKind, EM_X86_64};
  if (S == "aarch64linux")
    return {ELF64LEKind, EM_AARCH64};
  if (S == "i386pe" || S == "i386pep" || S == "thumb2pe")
    error("Windows targets are not supported on the ELF frontend: " + S);
  error("Unknown emulation: " + S);
}

// Returns slices of MB by parsing MB as an archive file.
// Each slice consists of a member file in the archive.
static std::vector<MemoryBufferRef> getArchiveMembers(MemoryBufferRef MB) {
  ErrorOr<std::unique_ptr<Archive>> FileOrErr = Archive::create(MB);
  error(FileOrErr, "Failed to parse archive");
  std::unique_ptr<Archive> File = std::move(*FileOrErr);

  std::vector<MemoryBufferRef> V;
  for (const ErrorOr<Archive::Child> &C : File->children()) {
    error(C, "Could not get the child of the archive " + File->getFileName());
    ErrorOr<MemoryBufferRef> MbOrErr = C->getMemoryBufferRef();
    error(MbOrErr, "Could not get the buffer for a child of the archive " +
                       File->getFileName());
    V.push_back(*MbOrErr);
  }
  return V;
}

// Opens and parses a file. Path has to be resolved already.
// Newly created memory buffers are owned by this driver.
void LinkerDriver::addFile(StringRef Path) {
  using namespace llvm::sys::fs;
  if (Config->Verbose)
    llvm::outs() << Path << "\n";
  auto MBOrErr = MemoryBuffer::getFile(Path);
  error(MBOrErr, "cannot open " + Path);
  std::unique_ptr<MemoryBuffer> &MB = *MBOrErr;
  MemoryBufferRef MBRef = MB->getMemBufferRef();
  OwningMBs.push_back(std::move(MB)); // take MB ownership

  switch (identify_magic(MBRef.getBuffer())) {
  case file_magic::unknown:
    readLinkerScript(&Alloc, MBRef);
    return;
  case file_magic::archive:
    if (WholeArchive) {
      for (MemoryBufferRef MB : getArchiveMembers(MBRef))
        Files.push_back(createObjectFile(MB));
      return;
    }
    Files.push_back(make_unique<ArchiveFile>(MBRef));
    return;
  case file_magic::elf_shared_object:
    Files.push_back(createSharedFile(MBRef));
    return;
  default:
    Files.push_back(createObjectFile(MBRef));
  }
}

// Some command line options or some combinations of them are not allowed.
// This function checks for such errors.
static void checkOptions(opt::InputArgList &Args) {
  // Traditional linkers can generate re-linkable object files instead
  // of executables or DSOs. We don't support that since the feature
  // does not seem to provide more value than the static archiver.
  if (Args.hasArg(OPT_relocatable))
    error("-r option is not supported. Use 'ar' command instead.");

  // The MIPS ABI as of 2016 does not support the GNU-style symbol lookup
  // table which is a relatively new feature.
  if (Config->EMachine == EM_MIPS && Config->GnuHash)
    error("The .gnu.hash section is not compatible with the MIPS target.");

  if (Config->EMachine == EM_AMDGPU && !Config->Entry.empty())
    error("-e option is not valid for AMDGPU.");
}

static StringRef
getString(opt::InputArgList &Args, unsigned Key, StringRef Default = "") {
  if (auto *Arg = Args.getLastArg(Key))
    return Arg->getValue();
  return Default;
}

static bool hasZOption(opt::InputArgList &Args, StringRef Key) {
  for (auto *Arg : Args.filtered(OPT_z))
    if (Key == Arg->getValue())
      return true;
  return false;
}

void LinkerDriver::main(ArrayRef<const char *> ArgsArr) {
  initSymbols();

  opt::InputArgList Args = parseArgs(&Alloc, ArgsArr);
  readConfigs(Args);
  createFiles(Args);
  checkOptions(Args);

  switch (Config->EKind) {
  case ELF32LEKind:
    link<ELF32LE>(Args);
    return;
  case ELF32BEKind:
    link<ELF32BE>(Args);
    return;
  case ELF64LEKind:
    link<ELF64LE>(Args);
    return;
  case ELF64BEKind:
    link<ELF64BE>(Args);
    return;
  default:
    error("-m or at least a .o file required");
  }
}

// Initializes Config members by the command line options.
void LinkerDriver::readConfigs(opt::InputArgList &Args) {
  for (auto *Arg : Args.filtered(OPT_L))
    Config->SearchPaths.push_back(Arg->getValue());

  std::vector<StringRef> RPaths;
  for (auto *Arg : Args.filtered(OPT_rpath))
    RPaths.push_back(Arg->getValue());
  if (!RPaths.empty())
    Config->RPath = llvm::join(RPaths.begin(), RPaths.end(), ":");

  if (auto *Arg = Args.getLastArg(OPT_m)) {
    // Parse ELF{32,64}{LE,BE} and CPU type.
    StringRef S = Arg->getValue();
    std::tie(Config->EKind, Config->EMachine) = parseEmulation(S);
    Config->Emulation = S;
  }

  Config->AllowMultipleDefinition = Args.hasArg(OPT_allow_multiple_definition);
  Config->Bsymbolic = Args.hasArg(OPT_Bsymbolic);
  Config->DiscardAll = Args.hasArg(OPT_discard_all);
  Config->DiscardLocals = Args.hasArg(OPT_discard_locals);
  Config->DiscardNone = Args.hasArg(OPT_discard_none);
  Config->EnableNewDtags = !Args.hasArg(OPT_disable_new_dtags);
  Config->ExportDynamic = Args.hasArg(OPT_export_dynamic);
  Config->GcSections = Args.hasArg(OPT_gc_sections);
  Config->NoInhibitExec = Args.hasArg(OPT_noinhibit_exec);
  Config->NoUndefined = Args.hasArg(OPT_no_undefined);
  Config->PrintGcSections = Args.hasArg(OPT_print_gc_sections);
  Config->Shared = Args.hasArg(OPT_shared);
  Config->StripAll = Args.hasArg(OPT_strip_all);
  Config->Verbose = Args.hasArg(OPT_verbose);

  Config->DynamicLinker = getString(Args, OPT_dynamic_linker);
  Config->Entry = getString(Args, OPT_entry);
  Config->Fini = getString(Args, OPT_fini, "_fini");
  Config->Init = getString(Args, OPT_init, "_init");
  Config->OutputFile = getString(Args, OPT_o);
  Config->SoName = getString(Args, OPT_soname);
  Config->Sysroot = getString(Args, OPT_sysroot);

  Config->ZExecStack = hasZOption(Args, "execstack");
  Config->ZNodelete = hasZOption(Args, "nodelete");
  Config->ZNow = hasZOption(Args, "now");
  Config->ZOrigin = hasZOption(Args, "origin");
  Config->ZRelro = !hasZOption(Args, "norelro");

  if (auto *Arg = Args.getLastArg(OPT_O)) {
    StringRef Val = Arg->getValue();
    if (Val.getAsInteger(10, Config->Optimize))
      error("Invalid optimization level");
  }

  if (auto *Arg = Args.getLastArg(OPT_hash_style)) {
    StringRef S = Arg->getValue();
    if (S == "gnu") {
      Config->GnuHash = true;
      Config->SysvHash = false;
    } else if (S == "both") {
      Config->GnuHash = true;
    } else if (S != "sysv")
      error("Unknown hash style: " + S);
  }

  for (auto *Arg : Args.filtered(OPT_undefined))
    Config->Undefined.push_back(Arg->getValue());
}

void LinkerDriver::createFiles(opt::InputArgList &Args) {
  for (auto *Arg : Args) {
    switch (Arg->getOption().getID()) {
    case OPT_l:
      addFile(searchLibrary(Arg->getValue()));
      break;
    case OPT_INPUT:
    case OPT_script:
      addFile(Arg->getValue());
      break;
    case OPT_as_needed:
      Config->AsNeeded = true;
      break;
    case OPT_no_as_needed:
      Config->AsNeeded = false;
      break;
    case OPT_Bstatic:
      Config->Static = true;
      break;
    case OPT_Bdynamic:
      Config->Static = false;
      break;
    case OPT_whole_archive:
      WholeArchive = true;
      break;
    case OPT_no_whole_archive:
      WholeArchive = false;
      break;
    }
  }

  if (Files.empty())
    error("no input files.");
}

template <class ELFT> void LinkerDriver::link(opt::InputArgList &Args) {
  SymbolTable<ELFT> Symtab;
  Target.reset(createTarget());

  if (!Config->Shared) {
    // Add entry symbol.
    //
    // There is no entry symbol for AMDGPU binaries, so skip adding one to avoid
    // having and undefined symbol.
    if (Config->Entry.empty() && Config->EMachine != EM_AMDGPU)
      Config->Entry = (Config->EMachine == EM_MIPS) ? "__start" : "_start";

    // In the assembly for 32 bit x86 the _GLOBAL_OFFSET_TABLE_ symbol
    // is magical and is used to produce a R_386_GOTPC relocation.
    // The R_386_GOTPC relocation value doesn't actually depend on the
    // symbol value, so it could use an index of STN_UNDEF which, according
    // to the spec, means the symbol value is 0.
    // Unfortunately both gas and MC keep the _GLOBAL_OFFSET_TABLE_ symbol in
    // the object file.
    // The situation is even stranger on x86_64 where the assembly doesn't
    // need the magical symbol, but gas still puts _GLOBAL_OFFSET_TABLE_ as
    // an undefined symbol in the .o files.
    // Given that the symbol is effectively unused, we just create a dummy
    // hidden one to avoid the undefined symbol error.
    Symtab.addIgnored("_GLOBAL_OFFSET_TABLE_");
  }

  if (!Config->Entry.empty()) {
    // Set either EntryAddr (if S is a number) or EntrySym (otherwise).
    StringRef S = Config->Entry;
    if (S.getAsInteger(0, Config->EntryAddr))
      Config->EntrySym = Symtab.addUndefined(S);
  }

  if (Config->EMachine == EM_MIPS) {
    // On MIPS O32 ABI, _gp_disp is a magic symbol designates offset between
    // start of function and gp pointer into GOT. Use 'strong' variant of
    // the addIgnored to prevent '_gp_disp' substitution.
    Config->MipsGpDisp = Symtab.addIgnoredStrong("_gp_disp");

    // Define _gp for MIPS. st_value of _gp symbol will be updated by Writer
    // so that it points to an absolute address which is relative to GOT.
    // See "Global Data Symbols" in Chapter 6 in the following document:
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    Symtab.addAbsolute("_gp", ElfSym<ELFT>::MipsGp);
  }

  for (std::unique_ptr<InputFile> &F : Files)
    Symtab.addFile(std::move(F));

  for (StringRef S : Config->Undefined)
    Symtab.addUndefinedOpt(S);

  for (auto *Arg : Args.filtered(OPT_wrap))
    Symtab.wrap(Arg->getValue());

  if (Config->OutputFile.empty())
    Config->OutputFile = "a.out";

  // Write the result to the file.
  Symtab.scanShlibUndefined();
  if (Config->GcSections)
    markLive<ELFT>(&Symtab);
  writeResult<ELFT>(&Symtab);
}
