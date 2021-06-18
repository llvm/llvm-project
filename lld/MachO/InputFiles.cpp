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
#include "llvm/TextAPI/Architecture.h"
#include "llvm/TextAPI/InterfaceFile.h"

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

  // Multiple dylibs can be defined in one .tbd file.
  if (auto dylibFile = dyn_cast<DylibFile>(f))
    if (f->getName().endswith(".tbd"))
      return (f->getName() + "(" + dylibFile->dylibName + ")").str();

  if (f->archiveName.empty())
    return std::string(f->getName());
  return (f->archiveName + "(" + path::filename(f->getName()) + ")").str();
}

SetVector<InputFile *> macho::inputFiles;
std::unique_ptr<TarWriter> macho::tar;
int InputFile::idCount = 0;

static VersionTuple decodeVersion(uint32_t version) {
  unsigned major = version >> 16;
  unsigned minor = (version >> 8) & 0xffu;
  unsigned subMinor = version & 0xffu;
  return VersionTuple(major, minor, subMinor);
}

static std::vector<PlatformInfo> getPlatformInfos(const InputFile *input) {
  if (!isa<ObjFile>(input) && !isa<DylibFile>(input))
    return {};

  const char *hdr = input->mb.getBufferStart();

  std::vector<PlatformInfo> platformInfos;
  for (auto *cmd : findCommands<build_version_command>(hdr, LC_BUILD_VERSION)) {
    PlatformInfo info;
    info.target.Platform = static_cast<PlatformKind>(cmd->platform);
    info.minimum = decodeVersion(cmd->minos);
    platformInfos.emplace_back(std::move(info));
  }
  for (auto *cmd : findCommands<version_min_command>(
           hdr, LC_VERSION_MIN_MACOSX, LC_VERSION_MIN_IPHONEOS,
           LC_VERSION_MIN_TVOS, LC_VERSION_MIN_WATCHOS)) {
    PlatformInfo info;
    switch (cmd->cmd) {
    case LC_VERSION_MIN_MACOSX:
      info.target.Platform = PlatformKind::macOS;
      break;
    case LC_VERSION_MIN_IPHONEOS:
      info.target.Platform = PlatformKind::iOS;
      break;
    case LC_VERSION_MIN_TVOS:
      info.target.Platform = PlatformKind::tvOS;
      break;
    case LC_VERSION_MIN_WATCHOS:
      info.target.Platform = PlatformKind::watchOS;
      break;
    }
    info.minimum = decodeVersion(cmd->version);
    platformInfos.emplace_back(std::move(info));
  }

  return platformInfos;
}

static PlatformKind removeSimulator(PlatformKind platform) {
  // Mapping of platform to simulator and vice-versa.
  static const std::map<PlatformKind, PlatformKind> platformMap = {
      {PlatformKind::iOSSimulator, PlatformKind::iOS},
      {PlatformKind::tvOSSimulator, PlatformKind::tvOS},
      {PlatformKind::watchOSSimulator, PlatformKind::watchOS}};

  auto iter = platformMap.find(platform);
  if (iter == platformMap.end())
    return platform;
  return iter->second;
}

static bool checkCompatibility(const InputFile *input) {
  std::vector<PlatformInfo> platformInfos = getPlatformInfos(input);
  if (platformInfos.empty())
    return true;

  auto it = find_if(platformInfos, [&](const PlatformInfo &info) {
    return removeSimulator(info.target.Platform) ==
           removeSimulator(config->platform());
  });
  if (it == platformInfos.end()) {
    std::string platformNames;
    raw_string_ostream os(platformNames);
    interleave(
        platformInfos, os,
        [&](const PlatformInfo &info) {
          os << getPlatformName(info.target.Platform);
        },
        "/");
    error(toString(input) + " has platform " + platformNames +
          Twine(", which is different from target platform ") +
          getPlatformName(config->platform()));
    return false;
  }

  if (it->minimum <= config->platformInfo.minimum)
    return true;

  error(toString(input) + " has version " + it->minimum.getAsString() +
        ", which is newer than target minimum of " +
        config->platformInfo.minimum.getAsString());
  return false;
}

// Open a given file path and return it as a memory-mapped file.
Optional<MemoryBufferRef> macho::readFile(StringRef path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> mbOrErr = MemoryBuffer::getFile(path);
  if (std::error_code ec = mbOrErr.getError()) {
    error("cannot open " + path + ": " + ec.message());
    return None;
  }

  std::unique_ptr<MemoryBuffer> &mb = *mbOrErr;
  MemoryBufferRef mbref = mb->getMemBufferRef();
  make<std::unique_ptr<MemoryBuffer>>(std::move(mb)); // take mb ownership

  // If this is a regular non-fat file, return it.
  const char *buf = mbref.getBufferStart();
  const auto *hdr = reinterpret_cast<const fat_header *>(buf);
  if (mbref.getBufferSize() < sizeof(uint32_t) ||
      read32be(&hdr->magic) != FAT_MAGIC) {
    if (tar)
      tar->append(relativeToRoot(path), mbref.getBuffer());
    return mbref;
  }

  // Object files and archive files may be fat files, which contain multiple
  // real files for different CPU ISAs. Here, we search for a file that matches
  // with the current link target and returns it as a MemoryBufferRef.
  const auto *arch = reinterpret_cast<const fat_arch *>(buf + sizeof(*hdr));

  for (uint32_t i = 0, n = read32be(&hdr->nfat_arch); i < n; ++i) {
    if (reinterpret_cast<const char *>(arch + i + 1) >
        buf + mbref.getBufferSize()) {
      error(path + ": fat_arch struct extends beyond end of file");
      return None;
    }

    if (read32be(&arch[i].cputype) != static_cast<uint32_t>(target->cpuType) ||
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

InputFile::InputFile(Kind kind, const InterfaceFile &interface)
    : id(idCount++), fileKind(kind), name(saver.save(interface.getPath())) {}

template <class Section>
void ObjFile::parseSections(ArrayRef<Section> sections) {
  subsections.reserve(sections.size());
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());

  for (const Section &sec : sections) {
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
                                              uint64_t *offset) {
  auto it = std::prev(llvm::upper_bound(
      map, *offset, [](uint64_t value, SubsectionEntry subsecEntry) {
        return value < subsecEntry.offset;
      }));
  *offset -= it->offset;
  return it->isec;
}

template <class Section>
static bool validateRelocationInfo(InputFile *file, const Section &sec,
                                   relocation_info rel) {
  const RelocAttrs &relocAttrs = target->getRelocAttrs(rel.r_type);
  bool valid = true;
  auto message = [relocAttrs, file, sec, rel, &valid](const Twine &diagnostic) {
    valid = false;
    return (relocAttrs.name + " relocation " + diagnostic + " at offset " +
            std::to_string(rel.r_address) + " of " + sec.segname + "," +
            sec.sectname + " in " + toString(file))
        .str();
  };

  if (!relocAttrs.hasAttr(RelocAttrBits::LOCAL) && !rel.r_extern)
    error(message("must be extern"));
  if (relocAttrs.hasAttr(RelocAttrBits::PCREL) != rel.r_pcrel)
    error(message(Twine("must ") + (rel.r_pcrel ? "not " : "") +
                  "be PC-relative"));
  if (isThreadLocalVariables(sec.flags) &&
      !relocAttrs.hasAttr(RelocAttrBits::UNSIGNED))
    error(message("not allowed in thread-local section, must be UNSIGNED"));
  if (rel.r_length < 2 || rel.r_length > 3 ||
      !relocAttrs.hasAttr(static_cast<RelocAttrBits>(1 << rel.r_length))) {
    static SmallVector<StringRef, 4> widths{"0", "4", "8", "4 or 8"};
    error(message("has width " + std::to_string(1 << rel.r_length) +
                  " bytes, but must be " +
                  widths[(static_cast<int>(relocAttrs.bits) >> 2) & 3] +
                  " bytes"));
  }
  return valid;
}

template <class Section>
void ObjFile::parseRelocations(ArrayRef<Section> sectionHeaders,
                               const Section &sec, SubsectionMap &subsecMap) {
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
    // datum for each is a symbolic address. The result is the offset
    // between two addresses.
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

    int64_t pairedAddend = 0;
    relocation_info relInfo = relInfos[i];
    if (target->hasAttr(relInfo.r_type, RelocAttrBits::ADDEND)) {
      pairedAddend = SignExtend64<24>(relInfo.r_symbolnum);
      relInfo = relInfos[++i];
    }
    assert(i < relInfos.size());
    if (!validateRelocationInfo(this, sec, relInfo))
      continue;
    if (relInfo.r_address & R_SCATTERED)
      fatal("TODO: Scattered relocations not supported");

    bool isSubtrahend =
        target->hasAttr(relInfo.r_type, RelocAttrBits::SUBTRAHEND);
    int64_t embeddedAddend = target->getEmbeddedAddend(mb, sec.offset, relInfo);
    assert(!(embeddedAddend && pairedAddend));
    int64_t totalAddend = pairedAddend + embeddedAddend;
    Reloc r;
    r.type = relInfo.r_type;
    r.pcrel = relInfo.r_pcrel;
    r.length = relInfo.r_length;
    r.offset = relInfo.r_address;
    if (relInfo.r_extern) {
      r.referent = symbols[relInfo.r_symbolnum];
      r.addend = isSubtrahend ? 0 : totalAddend;
    } else {
      assert(!isSubtrahend);
      const Section &referentSec = sectionHeaders[relInfo.r_symbolnum - 1];
      uint64_t referentOffset;
      if (relInfo.r_pcrel) {
        // The implicit addend for pcrel section relocations is the pcrel offset
        // in terms of the addresses in the input file. Here we adjust it so
        // that it describes the offset from the start of the referent section.
        // FIXME This logic was written around x86_64 behavior -- ARM64 doesn't
        // have pcrel section relocations. We may want to factor this out into
        // the arch-specific .cpp file.
        assert(target->hasAttr(r.type, RelocAttrBits::BYTE4));
        referentOffset =
            sec.addr + relInfo.r_address + 4 + totalAddend - referentSec.addr;
      } else {
        // The addend for a non-pcrel relocation is its absolute address.
        referentOffset = totalAddend - referentSec.addr;
      }
      SubsectionMap &referentSubsecMap = subsections[relInfo.r_symbolnum - 1];
      r.referent = findContainingSubsection(referentSubsecMap, &referentOffset);
      r.addend = referentOffset;
    }

    InputSection *subsec = findContainingSubsection(subsecMap, &r.offset);
    subsec->relocs.push_back(r);

    if (isSubtrahend) {
      relocation_info minuendInfo = relInfos[++i];
      // SUBTRACTOR relocations should always be followed by an UNSIGNED one
      // attached to the same address.
      assert(target->hasAttr(minuendInfo.r_type, RelocAttrBits::UNSIGNED) &&
             relInfo.r_address == minuendInfo.r_address);
      Reloc p;
      p.type = minuendInfo.r_type;
      if (minuendInfo.r_extern) {
        p.referent = symbols[minuendInfo.r_symbolnum];
        p.addend = totalAddend;
      } else {
        uint64_t referentOffset =
            totalAddend - sectionHeaders[minuendInfo.r_symbolnum - 1].addr;
        SubsectionMap &referentSubsecMap =
            subsections[minuendInfo.r_symbolnum - 1];
        p.referent =
            findContainingSubsection(referentSubsecMap, &referentOffset);
        p.addend = referentOffset;
      }
      subsec->relocs.push_back(p);
    }
  }
}

template <class NList>
static macho::Symbol *createDefined(const NList &sym, StringRef name,
                                    InputSection *isec, uint64_t value,
                                    uint64_t size) {
  // Symbol scope is determined by sym.n_type & (N_EXT | N_PEXT):
  // N_EXT: Global symbols. These go in the symbol table during the link,
  //        and also in the export table of the output so that the dynamic
  //        linker sees them.
  // N_EXT | N_PEXT: Linkage unit (think: dylib) scoped. These go in the
  //                 symbol table during the link so that duplicates are
  //                 either reported (for non-weak symbols) or merged
  //                 (for weak symbols), but they do not go in the export
  //                 table of the output.
  // N_PEXT: Does not occur in input files in practice,
  //         a private extern must be external.
  // 0: Translation-unit scoped. These are not in the symbol table during
  //    link, and not in the export table of the output either.

  bool isWeakDefCanBeHidden =
      (sym.n_desc & (N_WEAK_DEF | N_WEAK_REF)) == (N_WEAK_DEF | N_WEAK_REF);

  if (sym.n_type & (N_EXT | N_PEXT)) {
    assert((sym.n_type & N_EXT) && "invalid input");
    bool isPrivateExtern = sym.n_type & N_PEXT;

    // lld's behavior for merging symbols is slightly different from ld64:
    // ld64 picks the winning symbol based on several criteria (see
    // pickBetweenRegularAtoms() in ld64's SymbolTable.cpp), while lld
    // just merges metadata and keeps the contents of the first symbol
    // with that name (see SymbolTable::addDefined). For:
    // * inline function F in a TU built with -fvisibility-inlines-hidden
    // * and inline function F in another TU built without that flag
    // ld64 will pick the one from the file built without
    // -fvisibility-inlines-hidden.
    // lld will instead pick the one listed first on the link command line and
    // give it visibility as if the function was built without
    // -fvisibility-inlines-hidden.
    // If both functions have the same contents, this will have the same
    // behavior. If not, it won't, but the input had an ODR violation in
    // that case.
    //
    // Similarly, merging a symbol
    // that's isPrivateExtern and not isWeakDefCanBeHidden with one
    // that's not isPrivateExtern but isWeakDefCanBeHidden technically
    // should produce one
    // that's not isPrivateExtern but isWeakDefCanBeHidden. That matters
    // with ld64's semantics, because it means the non-private-extern
    // definition will continue to take priority if more private extern
    // definitions are encountered. With lld's semantics there's no observable
    // difference between a symbol that's isWeakDefCanBeHidden or one that's
    // privateExtern -- neither makes it into the dynamic symbol table. So just
    // promote isWeakDefCanBeHidden to isPrivateExtern here.
    if (isWeakDefCanBeHidden)
      isPrivateExtern = true;

    return symtab->addDefined(
        name, isec->file, isec, value, size, sym.n_desc & N_WEAK_DEF,
        isPrivateExtern, sym.n_desc & N_ARM_THUMB_DEF,
        sym.n_desc & REFERENCED_DYNAMICALLY, sym.n_desc & N_NO_DEAD_STRIP);
  }

  assert(!isWeakDefCanBeHidden &&
         "weak_def_can_be_hidden on already-hidden symbol?");
  return make<Defined>(
      name, isec->file, isec, value, size, sym.n_desc & N_WEAK_DEF,
      /*isExternal=*/false, /*isPrivateExtern=*/false,
      sym.n_desc & N_ARM_THUMB_DEF, sym.n_desc & REFERENCED_DYNAMICALLY,
      sym.n_desc & N_NO_DEAD_STRIP);
}

// Absolute symbols are defined symbols that do not have an associated
// InputSection. They cannot be weak.
template <class NList>
static macho::Symbol *createAbsolute(const NList &sym, InputFile *file,
                                     StringRef name) {
  if (sym.n_type & (N_EXT | N_PEXT)) {
    assert((sym.n_type & N_EXT) && "invalid input");
    return symtab->addDefined(name, file, nullptr, sym.n_value, /*size=*/0,
                              /*isWeakDef=*/false, sym.n_type & N_PEXT,
                              sym.n_desc & N_ARM_THUMB_DEF,
                              /*isReferencedDynamically=*/false,
                              sym.n_desc & N_NO_DEAD_STRIP);
  }
  return make<Defined>(name, file, nullptr, sym.n_value, /*size=*/0,
                       /*isWeakDef=*/false,
                       /*isExternal=*/false, /*isPrivateExtern=*/false,
                       sym.n_desc & N_ARM_THUMB_DEF,
                       /*isReferencedDynamically=*/false,
                       sym.n_desc & N_NO_DEAD_STRIP);
}

template <class NList>
macho::Symbol *ObjFile::parseNonSectionSymbol(const NList &sym,
                                              StringRef name) {
  uint8_t type = sym.n_type & N_TYPE;
  switch (type) {
  case N_UNDF:
    return sym.n_value == 0
               ? symtab->addUndefined(name, this, sym.n_desc & N_WEAK_REF)
               : symtab->addCommon(name, this, sym.n_value,
                                   1 << GET_COMM_ALIGN(sym.n_desc),
                                   sym.n_type & N_PEXT);
  case N_ABS:
    return createAbsolute(sym, this, name);
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

template <class LP>
void ObjFile::parseSymbols(ArrayRef<typename LP::section> sectionHeaders,
                           ArrayRef<typename LP::nlist> nList,
                           const char *strtab, bool subsectionsViaSymbols) {
  using NList = typename LP::nlist;

  // Groups indices of the symbols by the sections that contain them.
  std::vector<std::vector<uint32_t>> symbolsBySection(subsections.size());
  symbols.resize(nList.size());
  for (uint32_t i = 0; i < nList.size(); ++i) {
    const NList &sym = nList[i];
    StringRef name = strtab + sym.n_strx;
    if ((sym.n_type & N_TYPE) == N_SECT) {
      SubsectionMap &subsecMap = subsections[sym.n_sect - 1];
      // parseSections() may have chosen not to parse this section.
      if (subsecMap.empty())
        continue;
      symbolsBySection[sym.n_sect - 1].push_back(i);
    } else {
      symbols[i] = parseNonSectionSymbol(sym, name);
    }
  }

  // Calculate symbol sizes and create subsections by splitting the sections
  // along symbol boundaries.
  for (size_t i = 0; i < subsections.size(); ++i) {
    SubsectionMap &subsecMap = subsections[i];
    if (subsecMap.empty())
      continue;

    std::vector<uint32_t> &symbolIndices = symbolsBySection[i];
    llvm::sort(symbolIndices, [&](uint32_t lhs, uint32_t rhs) {
      return nList[lhs].n_value < nList[rhs].n_value;
    });
    uint64_t sectionAddr = sectionHeaders[i].addr;
    uint32_t sectionAlign = 1u << sectionHeaders[i].align;

    // We populate subsecMap by repeatedly splitting the last (highest address)
    // subsection.
    SubsectionEntry subsecEntry = subsecMap.back();
    for (size_t j = 0; j < symbolIndices.size(); ++j) {
      uint32_t symIndex = symbolIndices[j];
      const NList &sym = nList[symIndex];
      StringRef name = strtab + sym.n_strx;
      InputSection *isec = subsecEntry.isec;

      uint64_t subsecAddr = sectionAddr + subsecEntry.offset;
      uint64_t symbolOffset = sym.n_value - subsecAddr;
      uint64_t symbolSize =
          j + 1 < symbolIndices.size()
              ? nList[symbolIndices[j + 1]].n_value - sym.n_value
              : isec->data.size() - symbolOffset;
      // There are 3 cases where we do not need to create a new subsection:
      //   1. If the input file does not use subsections-via-symbols.
      //   2. Multiple symbols at the same address only induce one subsection.
      //      (The symbolOffset == 0 check covers both this case as well as
      //      the first loop iteration.)
      //   3. Alternative entry points do not induce new subsections.
      if (!subsectionsViaSymbols || symbolOffset == 0 ||
          sym.n_desc & N_ALT_ENTRY) {
        symbols[symIndex] =
            createDefined(sym, name, isec, symbolOffset, symbolSize);
        continue;
      }

      auto *nextIsec = make<InputSection>(*isec);
      nextIsec->data = isec->data.slice(symbolOffset);
      nextIsec->numRefs = 0;
      nextIsec->wasCoalesced = false;
      isec->data = isec->data.slice(0, symbolOffset);

      // By construction, the symbol will be at offset zero in the new
      // subsection.
      symbols[symIndex] =
          createDefined(sym, name, nextIsec, /*value=*/0, symbolSize);
      // TODO: ld64 appears to preserve the original alignment as well as each
      // subsection's offset from the last aligned address. We should consider
      // emulating that behavior.
      nextIsec->align = MinAlign(sectionAlign, sym.n_value);
      subsecMap.push_back({sym.n_value - sectionAddr, nextIsec});
      subsecEntry = subsecMap.back();
    }
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
  isec->live = true;
  subsections.push_back({{0, isec}});
}

ObjFile::ObjFile(MemoryBufferRef mb, uint32_t modTime, StringRef archiveName)
    : InputFile(ObjKind, mb), modTime(modTime) {
  this->archiveName = std::string(archiveName);
  if (target->wordSize == 8)
    parse<LP64>();
  else
    parse<ILP32>();
}

template <class LP> void ObjFile::parse() {
  using Header = typename LP::mach_header;
  using SegmentCommand = typename LP::segment_command;
  using Section = typename LP::section;
  using NList = typename LP::nlist;

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const Header *>(mb.getBufferStart());

  Architecture arch = getArchitectureFromCpuType(hdr->cputype, hdr->cpusubtype);
  if (arch != config->arch()) {
    error(toString(this) + " has architecture " + getArchitectureName(arch) +
          " which is incompatible with target architecture " +
          getArchitectureName(config->arch()));
    return;
  }

  if (!checkCompatibility(this))
    return;

  if (const load_command *cmd = findCommand(hdr, LC_LINKER_OPTION)) {
    auto *c = reinterpret_cast<const linker_option_command *>(cmd);
    StringRef data{reinterpret_cast<const char *>(c + 1),
                   c->cmdsize - sizeof(linker_option_command)};
    parseLCLinkerOption(this, c->count, data);
  }

  ArrayRef<Section> sectionHeaders;
  if (const load_command *cmd = findCommand(hdr, LP::segmentLCType)) {
    auto *c = reinterpret_cast<const SegmentCommand *>(cmd);
    sectionHeaders =
        ArrayRef<Section>{reinterpret_cast<const Section *>(c + 1), c->nsects};
    parseSections(sectionHeaders);
  }

  // TODO: Error on missing LC_SYMTAB?
  if (const load_command *cmd = findCommand(hdr, LC_SYMTAB)) {
    auto *c = reinterpret_cast<const symtab_command *>(cmd);
    ArrayRef<NList> nList(reinterpret_cast<const NList *>(buf + c->symoff),
                          c->nsyms);
    const char *strtab = reinterpret_cast<const char *>(buf) + c->stroff;
    bool subsectionsViaSymbols = hdr->flags & MH_SUBSECTIONS_VIA_SYMBOLS;
    parseSymbols<LP>(sectionHeaders, nList, strtab, subsectionsViaSymbols);
  }

  // The relocations may refer to the symbols, so we parse them after we have
  // parsed all the symbols.
  for (size_t i = 0, n = subsections.size(); i < n; ++i)
    if (!subsections[i].empty())
      parseRelocations(sectionHeaders, sectionHeaders[i], subsections[i]);

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
  // FIXME: There can be more than one compile unit per object file. See
  // PR48637.
  auto it = units.begin();
  compileUnit = it->get();
}

// The path can point to either a dylib or a .tbd file.
static DylibFile *loadDylib(StringRef path, DylibFile *umbrella) {
  Optional<MemoryBufferRef> mbref = readFile(path);
  if (!mbref) {
    error("could not read dylib file at " + path);
    return nullptr;
  }
  return loadDylib(*mbref, umbrella);
}

// TBD files are parsed into a series of TAPI documents (InterfaceFiles), with
// the first document storing child pointers to the rest of them. When we are
// processing a given TBD file, we store that top-level document in
// currentTopLevelTapi. When processing re-exports, we search its children for
// potentially matching documents in the same TBD file. Note that the children
// themselves don't point to further documents, i.e. this is a two-level tree.
//
// Re-exports can either refer to on-disk files, or to documents within .tbd
// files.
DylibFile *findDylib(StringRef path, DylibFile *umbrella,
                     const InterfaceFile *currentTopLevelTapi) {
  if (path::is_absolute(path, path::Style::posix))
    for (StringRef root : config->systemLibraryRoots)
      if (Optional<std::string> dylibPath =
              resolveDylibPath((root + path).str()))
        return loadDylib(*dylibPath, umbrella);

  // TODO: Expand @loader_path, @executable_path, @rpath etc, handle -dylib_path

  if (currentTopLevelTapi) {
    for (InterfaceFile &child :
         make_pointee_range(currentTopLevelTapi->documents())) {
      assert(child.documents().empty());
      if (path == child.getInstallName()) {
        auto file = make<DylibFile>(child, umbrella);
        file->parseReexports(child);
        return file;
      }
    }
  }

  if (Optional<std::string> dylibPath = resolveDylibPath(path))
    return loadDylib(*dylibPath, umbrella);

  return nullptr;
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

void loadReexport(StringRef path, DylibFile *umbrella,
                  const InterfaceFile *currentTopLevelTapi) {
  DylibFile *reexport = findDylib(path, umbrella, currentTopLevelTapi);
  if (!reexport)
    error("unable to locate re-export with install name " + path);
  else if (isImplicitlyLinked(path))
    inputFiles.insert(reexport);
}

DylibFile::DylibFile(MemoryBufferRef mb, DylibFile *umbrella,
                     bool isBundleLoader)
    : InputFile(DylibKind, mb), refState(RefState::Unreferenced),
      isBundleLoader(isBundleLoader) {
  assert(!isBundleLoader || !umbrella);
  if (umbrella == nullptr)
    umbrella = this;
  this->umbrella = umbrella;

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const mach_header *>(mb.getBufferStart());

  // Initialize dylibName.
  if (const load_command *cmd = findCommand(hdr, LC_ID_DYLIB)) {
    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    currentVersion = read32le(&c->dylib.current_version);
    compatibilityVersion = read32le(&c->dylib.compatibility_version);
    dylibName = reinterpret_cast<const char *>(cmd) + read32le(&c->dylib.name);
  } else if (!isBundleLoader) {
    // macho_executable and macho_bundle don't have LC_ID_DYLIB,
    // so it's OK.
    error("dylib " + toString(this) + " missing LC_ID_DYLIB load command");
    return;
  }

  if (config->printEachFile)
    message(toString(this));

  deadStrippable = hdr->flags & MH_DEAD_STRIPPABLE_DYLIB;

  if (!checkCompatibility(this))
    return;

  // Initialize symbols.
  exportingFile = isImplicitlyLinked(dylibName) ? this : this->umbrella;
  if (const load_command *cmd = findCommand(hdr, LC_DYLD_INFO_ONLY)) {
    auto *c = reinterpret_cast<const dyld_info_command *>(cmd);
    parseTrie(buf + c->export_off, c->export_size,
              [&](const Twine &name, uint64_t flags) {
                StringRef savedName = saver.save(name);
                if (handleLDSymbol(savedName))
                  return;
                bool isWeakDef = flags & EXPORT_SYMBOL_FLAGS_WEAK_DEFINITION;
                bool isTlv = flags & EXPORT_SYMBOL_FLAGS_KIND_THREAD_LOCAL;
                symbols.push_back(symtab->addDylib(savedName, exportingFile,
                                                   isWeakDef, isTlv));
              });
  } else {
    error("LC_DYLD_INFO_ONLY not found in " + toString(this));
    return;
  }
}

void DylibFile::parseLoadCommands(MemoryBufferRef mb) {
  auto *hdr = reinterpret_cast<const mach_header *>(mb.getBufferStart());
  const uint8_t *p = reinterpret_cast<const uint8_t *>(mb.getBufferStart()) +
                     target->headerSize;
  for (uint32_t i = 0, n = hdr->ncmds; i < n; ++i) {
    auto *cmd = reinterpret_cast<const load_command *>(p);
    p += cmd->cmdsize;

    if (!(hdr->flags & MH_NO_REEXPORTED_DYLIBS) &&
        cmd->cmd == LC_REEXPORT_DYLIB) {
      const auto *c = reinterpret_cast<const dylib_command *>(cmd);
      StringRef reexportPath =
          reinterpret_cast<const char *>(c) + read32le(&c->dylib.name);
      loadReexport(reexportPath, exportingFile, nullptr);
    }

    // FIXME: What about LC_LOAD_UPWARD_DYLIB, LC_LAZY_LOAD_DYLIB,
    // LC_LOAD_WEAK_DYLIB, LC_REEXPORT_DYLIB (..are reexports from dylibs with
    // MH_NO_REEXPORTED_DYLIBS loaded for -flat_namespace)?
    if (config->namespaceKind == NamespaceKind::flat &&
        cmd->cmd == LC_LOAD_DYLIB) {
      const auto *c = reinterpret_cast<const dylib_command *>(cmd);
      StringRef dylibPath =
          reinterpret_cast<const char *>(c) + read32le(&c->dylib.name);
      DylibFile *dylib = findDylib(dylibPath, umbrella, nullptr);
      if (!dylib)
        error(Twine("unable to locate library '") + dylibPath +
              "' loaded from '" + toString(this) + "' for -flat_namespace");
    }
  }
}

// Some versions of XCode ship with .tbd files that don't have the right
// platform settings.
static constexpr std::array<StringRef, 3> skipPlatformChecks{
    "/usr/lib/system/libsystem_kernel.dylib",
    "/usr/lib/system/libsystem_platform.dylib",
    "/usr/lib/system/libsystem_pthread.dylib"};

DylibFile::DylibFile(const InterfaceFile &interface, DylibFile *umbrella,
                     bool isBundleLoader)
    : InputFile(DylibKind, interface), refState(RefState::Unreferenced),
      isBundleLoader(isBundleLoader) {
  // FIXME: Add test for the missing TBD code path.

  if (umbrella == nullptr)
    umbrella = this;
  this->umbrella = umbrella;

  dylibName = saver.save(interface.getInstallName());
  compatibilityVersion = interface.getCompatibilityVersion().rawValue();
  currentVersion = interface.getCurrentVersion().rawValue();

  if (config->printEachFile)
    message(toString(this));

  if (!is_contained(skipPlatformChecks, dylibName) &&
      !is_contained(interface.targets(), config->platformInfo.target)) {
    error(toString(this) + " is incompatible with " +
          std::string(config->platformInfo.target));
    return;
  }

  exportingFile = isImplicitlyLinked(dylibName) ? this : umbrella;
  auto addSymbol = [&](const Twine &name) -> void {
    symbols.push_back(symtab->addDylib(saver.save(name), exportingFile,
                                       /*isWeakDef=*/false,
                                       /*isTlv=*/false));
  };
  // TODO(compnerd) filter out symbols based on the target platform
  // TODO: handle weak defs, thread locals
  for (const auto *symbol : interface.symbols()) {
    if (!symbol->getArchitectures().has(config->arch()))
      continue;

    if (handleLDSymbol(symbol->getName()))
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
}

void DylibFile::parseReexports(const InterfaceFile &interface) {
  const InterfaceFile *topLevel =
      interface.getParent() == nullptr ? &interface : interface.getParent();
  for (InterfaceFileRef intfRef : interface.reexportedLibraries()) {
    InterfaceFile::const_target_range targets = intfRef.targets();
    if (is_contained(skipPlatformChecks, intfRef.getInstallName()) ||
        is_contained(targets, config->platformInfo.target))
      loadReexport(intfRef.getInstallName(), exportingFile, topLevel);
  }
}

// $ld$ symbols modify the properties/behavior of the library (e.g. its install
// name, compatibility version or hide/add symbols) for specific target
// versions.
bool DylibFile::handleLDSymbol(StringRef originalName) {
  if (!originalName.startswith("$ld$"))
    return false;

  StringRef action;
  StringRef name;
  std::tie(action, name) = originalName.drop_front(strlen("$ld$")).split('$');
  if (action == "previous")
    handleLDPreviousSymbol(name, originalName);
  else if (action == "install_name")
    handleLDInstallNameSymbol(name, originalName);
  return true;
}

void DylibFile::handleLDPreviousSymbol(StringRef name, StringRef originalName) {
  // originalName: $ld$ previous $ <installname> $ <compatversion> $
  // <platformstr> $ <startversion> $ <endversion> $ <symbol-name> $
  StringRef installName;
  StringRef compatVersion;
  StringRef platformStr;
  StringRef startVersion;
  StringRef endVersion;
  StringRef symbolName;
  StringRef rest;

  std::tie(installName, name) = name.split('$');
  std::tie(compatVersion, name) = name.split('$');
  std::tie(platformStr, name) = name.split('$');
  std::tie(startVersion, name) = name.split('$');
  std::tie(endVersion, name) = name.split('$');
  std::tie(symbolName, rest) = name.split('$');
  // TODO: ld64 contains some logic for non-empty symbolName as well.
  if (!symbolName.empty())
    return;
  unsigned platform;
  if (platformStr.getAsInteger(10, platform) ||
      platform != static_cast<unsigned>(config->platform()))
    return;

  VersionTuple start;
  if (start.tryParse(startVersion)) {
    warn("failed to parse start version, symbol '" + originalName +
         "' ignored");
    return;
  }
  VersionTuple end;
  if (end.tryParse(endVersion)) {
    warn("failed to parse end version, symbol '" + originalName + "' ignored");
    return;
  }
  if (config->platformInfo.minimum < start ||
      config->platformInfo.minimum >= end)
    return;

  dylibName = saver.save(installName);

  if (!compatVersion.empty()) {
    VersionTuple cVersion;
    if (cVersion.tryParse(compatVersion)) {
      warn("failed to parse compatibility version, symbol '" + originalName +
           "' ignored");
      return;
    }
    compatibilityVersion = encodeVersion(cVersion);
  }
}

void DylibFile::handleLDInstallNameSymbol(StringRef name,
                                          StringRef originalName) {
  // originalName: $ld$ install_name $ os<version> $ install_name
  StringRef condition, installName;
  std::tie(condition, installName) = name.split('$');
  VersionTuple version;
  if (!condition.consume_front("os") || version.tryParse(condition))
    warn("failed to parse os version, symbol '" + originalName + "' ignored");
  else if (version == config->platformInfo.minimum)
    dylibName = saver.save(installName);
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

  // `sym` is owned by a LazySym, which will be replace<>()d by make<ObjFile>
  // and become invalid after that call. Copy it to the stack so we can refer
  // to it later.
  const object::Archive::Symbol symCopy = sym;

  if (Optional<InputFile *> file =
          loadArchiveMember(mb, modTime, getName(), /*objCOnly=*/false)) {
    inputFiles.insert(*file);
    // ld64 doesn't demangle sym here even with -demangle.
    // Match that: intentionally don't call toMachOString().
    printArchiveMemberLoad(symCopy.getName(), *file);
  }
}

static macho::Symbol *createBitcodeSymbol(const lto::InputFile::Symbol &objSym,
                                          BitcodeFile &file) {
  StringRef name = saver.save(objSym.getName());

  // TODO: support weak references
  if (objSym.isUndefined())
    return symtab->addUndefined(name, &file, /*isWeakRef=*/false);

  assert(!objSym.isCommon() && "TODO: support common symbols in LTO");

  // TODO: Write a test demonstrating why computing isPrivateExtern before
  // LTO compilation is important.
  bool isPrivateExtern = false;
  switch (objSym.getVisibility()) {
  case GlobalValue::HiddenVisibility:
    isPrivateExtern = true;
    break;
  case GlobalValue::ProtectedVisibility:
    error(name + " has protected visibility, which is not supported by Mach-O");
    break;
  case GlobalValue::DefaultVisibility:
    break;
  }

  return symtab->addDefined(name, &file, /*isec=*/nullptr, /*value=*/0,
                            /*size=*/0, objSym.isWeak(), isPrivateExtern,
                            /*isThumb=*/false,
                            /*isReferencedDynamically=*/false,
                            /*noDeadStrip=*/false);
}

BitcodeFile::BitcodeFile(MemoryBufferRef mbref)
    : InputFile(BitcodeKind, mbref) {
  obj = check(lto::InputFile::create(mbref));

  // Convert LTO Symbols to LLD Symbols in order to perform resolution. The
  // "winning" symbol will then be marked as Prevailing at LTO compilation
  // time.
  for (const lto::InputFile::Symbol &objSym : obj->symbols())
    symbols.push_back(createBitcodeSymbol(objSym, *this));
}

template void ObjFile::parse<LP64>();
