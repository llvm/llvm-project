//===- InputFiles.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains functions to parse Mach-O object files. In this comment,
// we describe the Mach-O file structure and how we parse it.
//
// Mach-O is not very different from ELF or COFF. The notion of symbols,
// sections and relocations exists in Mach-O as it does in ELF and COFF.
//
// Perhaps the notion that is new to those who know ELF/COFF is "subsections".
// In ELF/COFF, sections are an atomic unit of data copied from input files to
// output files. When we merge or garbage-collect sections, we treat each
// section as an atomic unit. In Mach-O, that's not the case. Sections can
// consist of multiple subsections, and subsections are a unit of merging and
// garbage-collecting. Therefore, Mach-O's subsections are more similar to
// ELF/COFF's sections than Mach-O's sections are.
//
// A section can have multiple symbols. A symbol that does not have the
// N_ALT_ENTRY attribute indicates a beginning of a subsection. Therefore, by
// definition, a symbol is always present at the beginning of each subsection. A
// symbol with N_ALT_ENTRY attribute does not start a new subsection and can
// point to a middle of a subsection.
//
// The notion of subsections also affects how relocations are represented in
// Mach-O. All references within a section need to be explicitly represented as
// relocations if they refer to different subsections, because we obviously need
// to fix up addresses if subsections are laid out in an output file differently
// than they were in object files. To represent that, Mach-O relocations can
// refer to an unnamed location via its address. Scattered relocations (those
// with the R_SCATTERED bit set) always refer to unnamed locations.
// Non-scattered relocations refer to an unnamed location if r_extern is not set
// and r_symbolnum is zero.
//
// Without the above differences, I think you can use your knowledge about ELF
// and COFF for Mach-O.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Config.h"
#include "Driver.h"
#include "Dwarf.h"
#include "ExportTrie.h"
#include "InputSection.h"
#include "MachOStructs.h"
#include "ObjC.h"
#include "OutputSection.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"

#include "lld/Common/DWARF.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Reproduce.h"
#include "llvm/ADT/iterator.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/LTO/LTO.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TarWriter.h"

using namespace llvm;
using namespace llvm::MachO;
using namespace llvm::support::endian;
using namespace llvm::sys;
using namespace lld;
using namespace lld::macho;

// Returns "<internal>", "foo.a(bar.o)", or "baz.o".
std::string lld::toString(const InputFile *f) {
  if (!f)
    return "<internal>";
  if (f->archiveName.empty())
    return std::string(f->getName());
  return (path::filename(f->archiveName) + "(" + path::filename(f->getName()) +
          ")")
      .str();
}

SetVector<InputFile *> macho::inputFiles;
std::unique_ptr<TarWriter> macho::tar;
int InputFile::idCount = 0;

// Open a given file path and return it as a memory-mapped file.
Optional<MemoryBufferRef> macho::readFile(StringRef path) {
  // Open a file.
  auto mbOrErr = MemoryBuffer::getFile(path);
  if (auto ec = mbOrErr.getError()) {
    error("cannot open " + path + ": " + ec.message());
    return None;
  }

  std::unique_ptr<MemoryBuffer> &mb = *mbOrErr;
  MemoryBufferRef mbref = mb->getMemBufferRef();
  make<std::unique_ptr<MemoryBuffer>>(std::move(mb)); // take mb ownership

  // If this is a regular non-fat file, return it.
  const char *buf = mbref.getBufferStart();
  auto *hdr = reinterpret_cast<const MachO::fat_header *>(buf);
  if (read32be(&hdr->magic) != MachO::FAT_MAGIC) {
    if (tar)
      tar->append(relativeToRoot(path), mbref.getBuffer());
    return mbref;
  }

  // Object files and archive files may be fat files, which contains
  // multiple real files for different CPU ISAs. Here, we search for a
  // file that matches with the current link target and returns it as
  // a MemoryBufferRef.
  auto *arch = reinterpret_cast<const MachO::fat_arch *>(buf + sizeof(*hdr));

  for (uint32_t i = 0, n = read32be(&hdr->nfat_arch); i < n; ++i) {
    if (reinterpret_cast<const char *>(arch + i + 1) >
        buf + mbref.getBufferSize()) {
      error(path + ": fat_arch struct extends beyond end of file");
      return None;
    }

    if (read32be(&arch[i].cputype) != target->cpuType ||
        read32be(&arch[i].cpusubtype) != target->cpuSubtype)
      continue;

    uint32_t offset = read32be(&arch[i].offset);
    uint32_t size = read32be(&arch[i].size);
    if (offset + size > mbref.getBufferSize())
      error(path + ": slice extends beyond end of file");
    if (tar)
      tar->append(relativeToRoot(path), mbref.getBuffer());
    return MemoryBufferRef(StringRef(buf + offset, size), path.copy(bAlloc));
  }

  error("unable to find matching architecture in " + path);
  return None;
}

const load_command *macho::findCommand(const mach_header_64 *hdr,
                                       uint32_t type) {
  const uint8_t *p =
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(mach_header_64);

  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const load_command *>(p);
    if (cmd->cmd == type)
      return cmd;
    p += cmd->cmdsize;
  }
  return nullptr;
}

void ObjFile::parseSections(ArrayRef<section_64> sections) {
  subsections.reserve(sections.size());
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());

  for (const section_64 &sec : sections) {
    InputSection *isec = make<InputSection>();
    isec->file = this;
    isec->name =
        StringRef(sec.sectname, strnlen(sec.sectname, sizeof(sec.sectname)));
    isec->segname =
        StringRef(sec.segname, strnlen(sec.segname, sizeof(sec.segname)));
    isec->data = {isZeroFill(sec.flags) ? nullptr : buf + sec.offset,
                  static_cast<size_t>(sec.size)};
    if (sec.align >= 32)
      error("alignment " + std::to_string(sec.align) + " of section " +
            isec->name + " is too large");
    else
      isec->align = 1 << sec.align;
    isec->flags = sec.flags;

    if (!(isDebugSection(isec->flags) &&
          isec->segname == segment_names::dwarf)) {
      subsections.push_back({{0, isec}});
    } else {
      // Instead of emitting DWARF sections, we emit STABS symbols to the
      // object files that contain them. We filter them out early to avoid
      // parsing their relocations unnecessarily. But we must still push an
      // empty map to ensure the indices line up for the remaining sections.
      subsections.push_back({});
      debugSections.push_back(isec);
    }
  }
}

// Find the subsection corresponding to the greatest section offset that is <=
// that of the given offset.
//
// offset: an offset relative to the start of the original InputSection (before
// any subsection splitting has occurred). It will be updated to represent the
// same location as an offset relative to the start of the containing
// subsection.
static InputSection *findContainingSubsection(SubsectionMap &map,
                                              uint32_t *offset) {
  auto it = std::prev(map.upper_bound(*offset));
  *offset -= it->first;
  return it->second;
}

void ObjFile::parseRelocations(const section_64 &sec,
                               SubsectionMap &subsecMap) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  ArrayRef<relocation_info> relInfos(
      reinterpret_cast<const relocation_info *>(buf + sec.reloff), sec.nreloc);

  for (size_t i = 0; i < relInfos.size(); i++) {
    // Paired relocations serve as Mach-O's method for attaching a
    // supplemental datum to a primary relocation record. ELF does not
    // need them because the *_RELOC_RELA records contain the extra
    // addend field, vs. *_RELOC_REL which omit the addend.
    //
    // The {X86_64,ARM64}_RELOC_SUBTRACTOR record holds the subtrahend,
    // and the paired *_RELOC_UNSIGNED record holds the minuend. The
    // datum for each is a symbolic address. The result is the runtime
    // offset between two addresses.
    //
    // The ARM64_RELOC_ADDEND record holds the addend, and the paired
    // ARM64_RELOC_BRANCH26 or ARM64_RELOC_PAGE21/PAGEOFF12 holds the
    // base symbolic address.
    //
    // Note: X86 does not use *_RELOC_ADDEND because it can embed an
    // addend into the instruction stream. On X86, a relocatable address
    // field always occupies an entire contiguous sequence of byte(s),
    // so there is no need to merge opcode bits with address
    // bits. Therefore, it's easy and convenient to store addends in the
    // instruction-stream bytes that would otherwise contain zeroes. By
    // contrast, RISC ISAs such as ARM64 mix opcode bits with with
    // address bits so that bitwise arithmetic is necessary to extract
    // and insert them. Storing addends in the instruction stream is
    // possible, but inconvenient and more costly at link time.

    relocation_info pairedInfo = relInfos[i];
    relocation_info relInfo =
        target->isPairedReloc(pairedInfo) ? relInfos[++i] : pairedInfo;
    assert(i < relInfos.size());
    if (relInfo.r_address & R_SCATTERED)
      fatal("TODO: Scattered relocations not supported");

    Reloc r;
    r.type = relInfo.r_type;
    r.pcrel = relInfo.r_pcrel;
    r.length = relInfo.r_length;
    r.offset = relInfo.r_address;
    // For unpaired relocs, pairdInfo (just a copy of relInfo) is ignored
    uint64_t rawAddend = target->getAddend(mb, sec, relInfo, pairedInfo);
    if (relInfo.r_extern) {
      r.referent = symbols[relInfo.r_symbolnum];
      r.addend = rawAddend;
    } else {
      SubsectionMap &referentSubsecMap = subsections[relInfo.r_symbolnum - 1];
      const section_64 &referentSec = sectionHeaders[relInfo.r_symbolnum - 1];
      uint32_t referentOffset;
      if (relInfo.r_pcrel) {
        // The implicit addend for pcrel section relocations is the pcrel offset
        // in terms of the addresses in the input file. Here we adjust it so
        // that it describes the offset from the start of the referent section.
        // TODO: The offset of 4 is probably not right for ARM64, nor for
        //       relocations with r_length != 2.
        referentOffset =
            sec.addr + relInfo.r_address + 4 + rawAddend - referentSec.addr;
      } else {
        // The addend for a non-pcrel relocation is its absolute address.
        referentOffset = rawAddend - referentSec.addr;
      }
      r.referent = findContainingSubsection(referentSubsecMap, &referentOffset);
      r.addend = referentOffset;
    }

    InputSection *subsec = findContainingSubsection(subsecMap, &r.offset);
    subsec->relocs.push_back(r);
  }
}

static macho::Symbol *createDefined(const structs::nlist_64 &sym,
                                    StringRef name, InputSection *isec,
                                    uint32_t value) {
  // Symbol scope is determined by sym.n_type & (N_EXT | N_PEXT):
  // N_EXT: Global symbols
  // N_EXT | N_PEXT: Linkage unit (think: dylib) scoped
  // N_PEXT: Does not occur in input files in practice,
  //         a private extern must be external.
  // 0: Translation-unit scoped. These are not in the symbol table.

  if (sym.n_type & (N_EXT | N_PEXT)) {
    assert((sym.n_type & N_EXT) && "invalid input");
    return symtab->addDefined(name, isec, value, sym.n_desc & N_WEAK_DEF,
                              sym.n_type & N_PEXT);
  }
  return make<Defined>(name, isec, value, sym.n_desc & N_WEAK_DEF,
                       /*isExternal=*/false, /*isPrivateExtern=*/false);
}

// Absolute symbols are defined symbols that do not have an associated
// InputSection. They cannot be weak.
static macho::Symbol *createAbsolute(const structs::nlist_64 &sym,
                                     StringRef name) {
  if (sym.n_type & (N_EXT | N_PEXT)) {
    assert((sym.n_type & N_EXT) && "invalid input");
    return symtab->addDefined(name, nullptr, sym.n_value, /*isWeakDef=*/false,
                              sym.n_type & N_PEXT);
  }
  return make<Defined>(name, nullptr, sym.n_value, /*isWeakDef=*/false,
                       /*isExternal=*/false, /*isPrivateExtern=*/false);
}

macho::Symbol *ObjFile::parseNonSectionSymbol(const structs::nlist_64 &sym,
                                              StringRef name) {
  uint8_t type = sym.n_type & N_TYPE;
  switch (type) {
  case N_UNDF:
    return sym.n_value == 0
               ? symtab->addUndefined(name, sym.n_desc & N_WEAK_REF)
               : symtab->addCommon(name, this, sym.n_value,
                                   1 << GET_COMM_ALIGN(sym.n_desc),
                                   sym.n_type & N_PEXT);
  case N_ABS:
    return createAbsolute(sym, name);
  case N_PBUD:
  case N_INDR:
    error("TODO: support symbols of type " + std::to_string(type));
    return nullptr;
  case N_SECT:
    llvm_unreachable(
        "N_SECT symbols should not be passed to parseNonSectionSymbol");
  default:
    llvm_unreachable("invalid symbol type");
  }
}

void ObjFile::parseSymbols(ArrayRef<structs::nlist_64> nList,
                           const char *strtab, bool subsectionsViaSymbols) {
  // resize(), not reserve(), because we are going to create N_ALT_ENTRY symbols
  // out-of-sequence.
  symbols.resize(nList.size());
  std::vector<size_t> altEntrySymIdxs;

  for (size_t i = 0, n = nList.size(); i < n; ++i) {
    const structs::nlist_64 &sym = nList[i];
    StringRef name = strtab + sym.n_strx;

    if ((sym.n_type & N_TYPE) != N_SECT) {
      symbols[i] = parseNonSectionSymbol(sym, name);
      continue;
    }

    const section_64 &sec = sectionHeaders[sym.n_sect - 1];
    SubsectionMap &subsecMap = subsections[sym.n_sect - 1];
    assert(!subsecMap.empty());
    uint64_t offset = sym.n_value - sec.addr;

    // If the input file does not use subsections-via-symbols, all symbols can
    // use the same subsection. Otherwise, we must split the sections along
    // symbol boundaries.
    if (!subsectionsViaSymbols) {
      symbols[i] = createDefined(sym, name, subsecMap[0], offset);
      continue;
    }

    // nList entries aren't necessarily arranged in address order. Therefore,
    // we can't create alt-entry symbols at this point because a later symbol
    // may split its section, which may affect which subsection the alt-entry
    // symbol is assigned to. So we need to handle them in a second pass below.
    if (sym.n_desc & N_ALT_ENTRY) {
      altEntrySymIdxs.push_back(i);
      continue;
    }

    // Find the subsection corresponding to the greatest section offset that is
    // <= that of the current symbol. The subsection that we find either needs
    // to be used directly or split in two.
    uint32_t firstSize = offset;
    InputSection *firstIsec = findContainingSubsection(subsecMap, &firstSize);

    if (firstSize == 0) {
      // Alias of an existing symbol, or the first symbol in the section. These
      // are handled by reusing the existing section.
      symbols[i] = createDefined(sym, name, firstIsec, 0);
      continue;
    }

    // We saw a symbol definition at a new offset. Split the section into two
    // subsections. The new symbol uses the second subsection.
    auto *secondIsec = make<InputSection>(*firstIsec);
    secondIsec->data = firstIsec->data.slice(firstSize);
    firstIsec->data = firstIsec->data.slice(0, firstSize);
    // TODO: ld64 appears to preserve the original alignment as well as each
    // subsection's offset from the last aligned address. We should consider
    // emulating that behavior.
    secondIsec->align = MinAlign(firstIsec->align, offset);

    subsecMap[offset] = secondIsec;
    // By construction, the symbol will be at offset zero in the new section.
    symbols[i] = createDefined(sym, name, secondIsec, 0);
  }

  for (size_t idx : altEntrySymIdxs) {
    const structs::nlist_64 &sym = nList[idx];
    StringRef name = strtab + sym.n_strx;
    SubsectionMap &subsecMap = subsections[sym.n_sect - 1];
    uint32_t off = sym.n_value - sectionHeaders[sym.n_sect - 1].addr;
    InputSection *subsec = findContainingSubsection(subsecMap, &off);
    symbols[idx] = createDefined(sym, name, subsec, off);
  }
}

OpaqueFile::OpaqueFile(MemoryBufferRef mb, StringRef segName,
                       StringRef sectName)
    : InputFile(OpaqueKind, mb) {
  InputSection *isec = make<InputSection>();
  isec->file = this;
  isec->name = sectName.take_front(16);
  isec->segname = segName.take_front(16);
  const auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  isec->data = {buf, mb.getBufferSize()};
  subsections.push_back({{0, isec}});
}

ObjFile::ObjFile(MemoryBufferRef mb, uint32_t modTime, StringRef archiveName)
    : InputFile(ObjKind, mb), modTime(modTime) {
  this->archiveName = std::string(archiveName);

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());

  if (const load_command *cmd = findCommand(hdr, LC_LINKER_OPTION)) {
    auto *c = reinterpret_cast<const linker_option_command *>(cmd);
    StringRef data{reinterpret_cast<const char *>(c + 1),
                   c->cmdsize - sizeof(linker_option_command)};
    parseLCLinkerOption(this, c->count, data);
  }

  if (const load_command *cmd = findCommand(hdr, LC_SEGMENT_64)) {
    auto *c = reinterpret_cast<const segment_command_64 *>(cmd);
    sectionHeaders = ArrayRef<section_64>{
        reinterpret_cast<const section_64 *>(c + 1), c->nsects};
    parseSections(sectionHeaders);
  }

  // TODO: Error on missing LC_SYMTAB?
  if (const load_command *cmd = findCommand(hdr, LC_SYMTAB)) {
    auto *c = reinterpret_cast<const symtab_command *>(cmd);
    ArrayRef<structs::nlist_64> nList(
        reinterpret_cast<const structs::nlist_64 *>(buf + c->symoff), c->nsyms);
    const char *strtab = reinterpret_cast<const char *>(buf) + c->stroff;
    bool subsectionsViaSymbols = hdr->flags & MH_SUBSECTIONS_VIA_SYMBOLS;
    parseSymbols(nList, strtab, subsectionsViaSymbols);
  }

  // The relocations may refer to the symbols, so we parse them after we have
  // parsed all the symbols.
  for (size_t i = 0, n = subsections.size(); i < n; ++i)
    if (!subsections[i].empty())
      parseRelocations(sectionHeaders[i], subsections[i]);

  parseDebugInfo();
}

void ObjFile::parseDebugInfo() {
  std::unique_ptr<DwarfObject> dObj = DwarfObject::create(this);
  if (!dObj)
    return;

  auto *ctx = make<DWARFContext>(
      std::move(dObj), "",
      [&](Error err) {
        warn(toString(this) + ": " + toString(std::move(err)));
      },
      [&](Error warning) {
        warn(toString(this) + ": " + toString(std::move(warning)));
      });

  // TODO: Since object files can contain a lot of DWARF info, we should verify
  // that we are parsing just the info we need
  const DWARFContext::compile_unit_range &units = ctx->compile_units();
  auto it = units.begin();
  compileUnit = it->get();
  assert(std::next(it) == units.end());
}

// The path can point to either a dylib or a .tbd file.
static Optional<DylibFile *> loadDylib(StringRef path, DylibFile *umbrella) {
  Optional<MemoryBufferRef> mbref = readFile(path);
  if (!mbref) {
    error("could not read dylib file at " + path);
    return {};
  }
  return loadDylib(*mbref, umbrella);
}

// TBD files are parsed into a series of TAPI documents (InterfaceFiles), with
// the first document storing child pointers to the rest of them. When we are
// processing a given TBD file, we store that top-level document here. When
// processing re-exports, we search its children for potentially matching
// documents in the same TBD file. Note that the children themselves don't
// point to further documents, i.e. this is a two-level tree.
//
// ld64 allows a TAPI re-export to reference documents nested within other TBD
// files, but that seems like a strange design, so this is an intentional
// deviation.
const InterfaceFile *currentTopLevelTapi = nullptr;

// Re-exports can either refer to on-disk files, or to documents within .tbd
// files.
static Optional<DylibFile *> loadReexportHelper(StringRef path,
                                                DylibFile *umbrella) {
  if (path::is_absolute(path, path::Style::posix))
    for (StringRef root : config->systemLibraryRoots)
      if (Optional<std::string> dylibPath =
              resolveDylibPath((root + path).str()))
        return loadDylib(*dylibPath, umbrella);

  // TODO: Expand @loader_path, @executable_path etc

  if (currentTopLevelTapi) {
    for (InterfaceFile &child :
         make_pointee_range(currentTopLevelTapi->documents())) {
      if (path == child.getInstallName())
        return make<DylibFile>(child, umbrella);
      assert(child.documents().empty());
    }
  }

  if (Optional<std::string> dylibPath = resolveDylibPath(path))
    return loadDylib(*dylibPath, umbrella);

  error("unable to locate re-export with install name " + path);
  return {};
}

// If a re-exported dylib is public (lives in /usr/lib or
// /System/Library/Frameworks), then it is considered implicitly linked: we
// should bind to its symbols directly instead of via the re-exporting umbrella
// library.
static bool isImplicitlyLinked(StringRef path) {
  if (!config->implicitDylibs)
    return false;

  if (path::parent_path(path) == "/usr/lib")
    return true;

  // Match /System/Library/Frameworks/$FOO.framework/**/$FOO
  if (path.consume_front("/System/Library/Frameworks/")) {
    StringRef frameworkName = path.take_until([](char c) { return c == '.'; });
    return path::filename(path) == frameworkName;
  }

  return false;
}

void loadReexport(StringRef path, DylibFile *umbrella) {
  Optional<DylibFile *> reexport = loadReexportHelper(path, umbrella);
  if (reexport && isImplicitlyLinked(path))
    inputFiles.insert(*reexport);
}

DylibFile::DylibFile(MemoryBufferRef mb, DylibFile *umbrella)
    : InputFile(DylibKind, mb), refState(RefState::Unreferenced) {
  if (umbrella == nullptr)
    umbrella = this;

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header_64 *>(mb.getBufferStart());

  // Initialize dylibName.
  if (const load_command *cmd = findCommand(hdr, LC_ID_DYLIB)) {
    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    currentVersion = read32le(&c->dylib.current_version);
    compatibilityVersion = read32le(&c->dylib.compatibility_version);
    dylibName = reinterpret_cast<const char *>(cmd) + read32le(&c->dylib.name);
  } else {
    error("dylib " + toString(this) + " missing LC_ID_DYLIB load command");
    return;
  }

  // Initialize symbols.
  DylibFile *exportingFile = isImplicitlyLinked(dylibName) ? this : umbrella;
  if (const load_command *cmd = findCommand(hdr, LC_DYLD_INFO_ONLY)) {
    auto *c = reinterpret_cast<const dyld_info_command *>(cmd);
    parseTrie(buf + c->export_off, c->export_size,
              [&](const Twine &name, uint64_t flags) {
                bool isWeakDef = flags & EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION;
                bool isTlv = flags & EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL;
                symbols.push_back(symtab->addDylib(
                    saver.save(name), exportingFile, isWeakDef, isTlv));
              });
  } else {
    error("LC_DYLD_INFO_ONLY not found in " + toString(this));
    return;
  }

  if (hdr->flags & MH_NO_REEXPORTED_DYLIBS)
    return;

  const uint8_t *p =
      reinterpret_cast<const uint8_t *>(hdr) + sizeof(mach_header_64);
  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const load_command *>(p);
    p += cmd->cmdsize;
    if (cmd->cmd != LC_REEXPORT_DYLIB)
      continue;

    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    StringRef reexportPath =
        reinterpret_cast<const char *>(c) + read32le(&c->dylib.name);
    loadReexport(reexportPath, umbrella);
  }
}

DylibFile::DylibFile(const InterfaceFile &interface, DylibFile *umbrella)
    : InputFile(DylibKind, interface), refState(RefState::Unreferenced) {
  if (umbrella == nullptr)
    umbrella = this;

  dylibName = saver.save(interface.getInstallName());
  compatibilityVersion = interface.getCompatibilityVersion().rawValue();
  currentVersion = interface.getCurrentVersion().rawValue();
  DylibFile *exportingFile = isImplicitlyLinked(dylibName) ? this : umbrella;
  auto addSymbol = [&](const Twine &name) -> void {
    symbols.push_back(symtab->addDylib(saver.save(name), exportingFile,
                                       /*isWeakDef=*/false,
                                       /*isTlv=*/false));
  };
  // TODO(compnerd) filter out symbols based on the target platform
  // TODO: handle weak defs, thread locals
  for (const auto symbol : interface.symbols()) {
    if (!symbol->getArchitectures().has(config->arch))
      continue;

    switch (symbol->getKind()) {
    case SymbolKind::GlobalSymbol:
      addSymbol(symbol->getName());
      break;
    case SymbolKind::ObjectiveCClass:
      // XXX ld64 only creates these symbols when -ObjC is passed in. We may
      // want to emulate that.
      addSymbol(objc::klass + symbol->getName());
      addSymbol(objc::metaclass + symbol->getName());
      break;
    case SymbolKind::ObjectiveCClassEHType:
      addSymbol(objc::ehtype + symbol->getName());
      break;
    case SymbolKind::ObjectiveCInstanceVariable:
      addSymbol(objc::ivar + symbol->getName());
      break;
    }
  }

  bool isTopLevelTapi = false;
  if (currentTopLevelTapi == nullptr) {
    currentTopLevelTapi = &interface;
    isTopLevelTapi = true;
  }

  for (InterfaceFileRef intfRef : interface.reexportedLibraries())
    loadReexport(intfRef.getInstallName(), umbrella);

  if (isTopLevelTapi)
    currentTopLevelTapi = nullptr;
}

ArchiveFile::ArchiveFile(std::unique_ptr<object::Archive> &&f)
    : InputFile(ArchiveKind, f->getMemoryBufferRef()), file(std::move(f)) {
  for (const object::Archive::Symbol &sym : file->symbols())
    symtab->addLazy(sym.getName(), this, sym);
}

void ArchiveFile::fetch(const object::Archive::Symbol &sym) {
  object::Archive::Child c =
      CHECK(sym.getMember(), toString(this) +
                                 ": could not get the member for symbol " +
                                 toMachOString(sym));

  if (!seen.insert(c.getChildOffset()).second)
    return;

  MemoryBufferRef mb =
      CHECK(c.getMemoryBufferRef(),
            toString(this) +
                ": could not get the buffer for the member defining symbol " +
                toMachOString(sym));

  if (tar && c.getParent()->isThin())
    tar->append(relativeToRoot(CHECK(c.getFullName(), this)), mb.getBuffer());

  uint32_t modTime = toTimeT(
      CHECK(c.getLastModified(), toString(this) +
                                     ": could not get the modification time "
                                     "for the member defining symbol " +
                                     toMachOString(sym)));

  // `sym` is owned by a LazySym, which will be replace<>() by make<ObjFile>
  // and become invalid after that call. Copy it to the stack so we can refer
  // to it later.
  const object::Archive::Symbol sym_copy = sym;

  InputFile *file;
  switch (identify_magic(mb.getBuffer())) {
  case file_magic::macho_object:
    file = make<ObjFile>(mb, modTime, getName());
    break;
  case file_magic::bitcode:
    file = make<BitcodeFile>(mb);
    break;
  default:
    StringRef bufname =
        CHECK(c.getName(), toString(this) + ": could not get buffer name");
    error(toString(this) + ": archive member " + bufname +
          " has unhandled file type");
    return;
  }
  inputFiles.insert(file);

  // ld64 doesn't demangle sym here even with -demangle. Match that, so
  // intentionally no call to toMachOString() here.
  printArchiveMemberLoad(sym_copy.getName(), file);
}

BitcodeFile::BitcodeFile(MemoryBufferRef mbref)
    : InputFile(BitcodeKind, mbref) {
  obj = check(lto::InputFile::create(mbref));
}
