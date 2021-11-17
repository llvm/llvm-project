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
#include "SyntheticSections.h"
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

#include <type_traits>

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
      return (f->getName() + "(" + dylibFile->installName + ")").str();

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

  if (it->minimum > config->platformInfo.minimum)
    warn(toString(input) + " has version " + it->minimum.getAsString() +
         ", which is newer than target minimum of " +
         config->platformInfo.minimum.getAsString());

  return true;
}

// This cache mostly exists to store system libraries (and .tbds) as they're
// loaded, rather than the input archives, which are already cached at a higher
// level, and other files like the filelist that are only read once.
// Theoretically this caching could be more efficient by hoisting it, but that
// would require altering many callers to track the state.
DenseMap<CachedHashStringRef, MemoryBufferRef> macho::cachedReads;
// Open a given file path and return it as a memory-mapped file.
Optional<MemoryBufferRef> macho::readFile(StringRef path) {
  CachedHashStringRef key(path);
  auto entry = cachedReads.find(key);
  if (entry != cachedReads.end())
    return entry->second;

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
    return cachedReads[key] = mbref;
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
    return cachedReads[key] = MemoryBufferRef(StringRef(buf + offset, size),
                                              path.copy(bAlloc));
  }

  error("unable to find matching architecture in " + path);
  return None;
}

InputFile::InputFile(Kind kind, const InterfaceFile &interface)
    : id(idCount++), fileKind(kind), name(saver.save(interface.getPath())) {}

// Some sections comprise of fixed-size records, so instead of splitting them at
// symbol boundaries, we split them based on size. Records are distinct from
// literals in that they may contain references to other sections, instead of
// being leaf nodes in the InputSection graph.
//
// Note that "record" is a term I came up with. In contrast, "literal" is a term
// used by the Mach-O format.
static Optional<size_t> getRecordSize(StringRef segname, StringRef name) {
  if (name == section_names::cfString) {
    if (config->icfLevel != ICFLevel::none && segname == segment_names::data)
      return target->wordSize == 8 ? 32 : 16;
  } else if (name == section_names::compactUnwind) {
    if (segname == segment_names::ld)
      return target->wordSize == 8 ? 32 : 20;
  }
  return {};
}

// Parse the sequence of sections within a single LC_SEGMENT(_64).
// Split each section into subsections.
template <class SectionHeader>
void ObjFile::parseSections(ArrayRef<SectionHeader> sectionHeaders) {
  sections.reserve(sectionHeaders.size());
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());

  for (const SectionHeader &sec : sectionHeaders) {
    StringRef name =
        StringRef(sec.sectname, strnlen(sec.sectname, sizeof(sec.sectname)));
    StringRef segname =
        StringRef(sec.segname, strnlen(sec.segname, sizeof(sec.segname)));
    ArrayRef<uint8_t> data = {isZeroFill(sec.flags) ? nullptr
                                                    : buf + sec.offset,
                              static_cast<size_t>(sec.size)};
    if (sec.align >= 32) {
      error("alignment " + std::to_string(sec.align) + " of section " + name +
            " is too large");
      sections.push_back(sec.addr);
      continue;
    }
    uint32_t align = 1 << sec.align;
    uint32_t flags = sec.flags;

    auto splitRecords = [&](int recordSize) -> void {
      sections.push_back(sec.addr);
      if (data.empty())
        return;
      Subsections &subsections = sections.back().subsections;
      subsections.reserve(data.size() / recordSize);
      auto *isec = make<ConcatInputSection>(
          segname, name, this, data.slice(0, recordSize), align, flags);
      subsections.push_back({0, isec});
      for (uint64_t off = recordSize; off < data.size(); off += recordSize) {
        // Copying requires less memory than constructing a fresh InputSection.
        auto *copy = make<ConcatInputSection>(*isec);
        copy->data = data.slice(off, recordSize);
        subsections.push_back({off, copy});
      }
    };

    if (sectionType(sec.flags) == S_CSTRING_LITERALS ||
        (config->dedupLiterals && isWordLiteralSection(sec.flags))) {
      if (sec.nreloc && config->dedupLiterals)
        fatal(toString(this) + " contains relocations in " + sec.segname + "," +
              sec.sectname +
              ", so LLD cannot deduplicate literals. Try re-running without "
              "--deduplicate-literals.");

      InputSection *isec;
      if (sectionType(sec.flags) == S_CSTRING_LITERALS) {
        isec =
            make<CStringInputSection>(segname, name, this, data, align, flags);
        // FIXME: parallelize this?
        cast<CStringInputSection>(isec)->splitIntoPieces();
      } else {
        isec = make<WordLiteralInputSection>(segname, name, this, data, align,
                                             flags);
      }
      sections.push_back(sec.addr);
      sections.back().subsections.push_back({0, isec});
    } else if (auto recordSize = getRecordSize(segname, name)) {
      splitRecords(*recordSize);
      if (name == section_names::compactUnwind)
        compactUnwindSection = &sections.back();
    } else if (segname == segment_names::llvm) {
      // ld64 does not appear to emit contents from sections within the __LLVM
      // segment. Symbols within those sections point to bitcode metadata
      // instead of actual symbols. Global symbols within those sections could
      // have the same name without causing duplicate symbol errors. Push an
      // empty entry to ensure indices line up for the remaining sections.
      // TODO: Evaluate whether the bitcode metadata is needed.
      sections.push_back(sec.addr);
    } else {
      auto *isec =
          make<ConcatInputSection>(segname, name, this, data, align, flags);
      if (isDebugSection(isec->getFlags()) &&
          isec->getSegName() == segment_names::dwarf) {
        // Instead of emitting DWARF sections, we emit STABS symbols to the
        // object files that contain them. We filter them out early to avoid
        // parsing their relocations unnecessarily. But we must still push an
        // empty entry to ensure the indices line up for the remaining sections.
        sections.push_back(sec.addr);
        debugSections.push_back(isec);
      } else {
        sections.push_back(sec.addr);
        sections.back().subsections.push_back({0, isec});
      }
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
template <class T>
static InputSection *findContainingSubsection(const Subsections &subsections,
                                              T *offset) {
  static_assert(std::is_same<uint64_t, T>::value ||
                    std::is_same<uint32_t, T>::value,
                "unexpected type for offset");
  auto it = std::prev(llvm::upper_bound(
      subsections, *offset,
      [](uint64_t value, Subsection subsec) { return value < subsec.offset; }));
  *offset -= it->offset;
  return it->isec;
}

template <class SectionHeader>
static bool validateRelocationInfo(InputFile *file, const SectionHeader &sec,
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

template <class SectionHeader>
void ObjFile::parseRelocations(ArrayRef<SectionHeader> sectionHeaders,
                               const SectionHeader &sec,
                               Subsections &subsections) {
  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  ArrayRef<relocation_info> relInfos(
      reinterpret_cast<const relocation_info *>(buf + sec.reloff), sec.nreloc);

  auto subsecIt = subsections.rbegin();
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

    relocation_info relInfo = relInfos[i];
    bool isSubtrahend =
        target->hasAttr(relInfo.r_type, RelocAttrBits::SUBTRAHEND);
    if (isSubtrahend && StringRef(sec.sectname) == section_names::ehFrame) {
      // __TEXT,__eh_frame only has symbols and SUBTRACTOR relocs when ld64 -r
      // adds local "EH_Frame1" and "func.eh". Ignore them because they have
      // gone unused by Mac OS since Snow Leopard (10.6), vintage 2009.
      ++i;
      continue;
    }
    int64_t pairedAddend = 0;
    if (target->hasAttr(relInfo.r_type, RelocAttrBits::ADDEND)) {
      pairedAddend = SignExtend64<24>(relInfo.r_symbolnum);
      relInfo = relInfos[++i];
    }
    assert(i < relInfos.size());
    if (!validateRelocationInfo(this, sec, relInfo))
      continue;
    if (relInfo.r_address & R_SCATTERED)
      fatal("TODO: Scattered relocations not supported");

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
      const SectionHeader &referentSecHead =
          sectionHeaders[relInfo.r_symbolnum - 1];
      uint64_t referentOffset;
      if (relInfo.r_pcrel) {
        // The implicit addend for pcrel section relocations is the pcrel offset
        // in terms of the addresses in the input file. Here we adjust it so
        // that it describes the offset from the start of the referent section.
        // FIXME This logic was written around x86_64 behavior -- ARM64 doesn't
        // have pcrel section relocations. We may want to factor this out into
        // the arch-specific .cpp file.
        assert(target->hasAttr(r.type, RelocAttrBits::BYTE4));
        referentOffset = sec.addr + relInfo.r_address + 4 + totalAddend -
                         referentSecHead.addr;
      } else {
        // The addend for a non-pcrel relocation is its absolute address.
        referentOffset = totalAddend - referentSecHead.addr;
      }
      Subsections &referentSubsections =
          sections[relInfo.r_symbolnum - 1].subsections;
      r.referent =
          findContainingSubsection(referentSubsections, &referentOffset);
      r.addend = referentOffset;
    }

    // Find the subsection that this relocation belongs to.
    // Though not required by the Mach-O format, clang and gcc seem to emit
    // relocations in order, so let's take advantage of it. However, ld64 emits
    // unsorted relocations (in `-r` mode), so we have a fallback for that
    // uncommon case.
    InputSection *subsec;
    while (subsecIt != subsections.rend() && subsecIt->offset > r.offset)
      ++subsecIt;
    if (subsecIt == subsections.rend() ||
        subsecIt->offset + subsecIt->isec->getSize() <= r.offset) {
      subsec = findContainingSubsection(subsections, &r.offset);
      // Now that we know the relocs are unsorted, avoid trying the 'fast path'
      // for the other relocations.
      subsecIt = subsections.rend();
    } else {
      subsec = subsecIt->isec;
      r.offset -= subsecIt->offset;
    }
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
        Subsections &referentSubsectVec =
            sections[minuendInfo.r_symbolnum - 1].subsections;
        p.referent =
            findContainingSubsection(referentSubsectVec, &referentOffset);
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
  // N_PEXT: llvm-mc does not emit these, but `ld -r` (wherein ld64 emits
  //         object files) may produce them. LLD does not yet support -r.
  //         These are translation-unit scoped, identical to the `0` case.
  // 0: Translation-unit scoped. These are not in the symbol table during
  //    link, and not in the export table of the output either.
  bool isWeakDefCanBeHidden =
      (sym.n_desc & (N_WEAK_DEF | N_WEAK_REF)) == (N_WEAK_DEF | N_WEAK_REF);

  if (sym.n_type & N_EXT) {
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
    // difference between a symbol that's isWeakDefCanBeHidden(autohide) or one
    // that's privateExtern -- neither makes it into the dynamic symbol table,
    // unless the autohide symbol is explicitly exported.
    // But if a symbol is both privateExtern and autohide then it can't
    // be exported.
    // So we nullify the autohide flag when privateExtern is present
    // and promote the symbol to privateExtern when it is not already.
    if (isWeakDefCanBeHidden && isPrivateExtern)
      isWeakDefCanBeHidden = false;
    else if (isWeakDefCanBeHidden)
      isPrivateExtern = true;
    return symtab->addDefined(
        name, isec->getFile(), isec, value, size, sym.n_desc & N_WEAK_DEF,
        isPrivateExtern, sym.n_desc & N_ARM_THUMB_DEF,
        sym.n_desc & REFERENCED_DYNAMICALLY, sym.n_desc & N_NO_DEAD_STRIP,
        isWeakDefCanBeHidden);
  }
  assert(!isWeakDefCanBeHidden &&
         "weak_def_can_be_hidden on already-hidden symbol?");
  return make<Defined>(
      name, isec->getFile(), isec, value, size, sym.n_desc & N_WEAK_DEF,
      /*isExternal=*/false, /*isPrivateExtern=*/false,
      sym.n_desc & N_ARM_THUMB_DEF, sym.n_desc & REFERENCED_DYNAMICALLY,
      sym.n_desc & N_NO_DEAD_STRIP);
}

// Absolute symbols are defined symbols that do not have an associated
// InputSection. They cannot be weak.
template <class NList>
static macho::Symbol *createAbsolute(const NList &sym, InputFile *file,
                                     StringRef name) {
  if (sym.n_type & N_EXT) {
    return symtab->addDefined(
        name, file, nullptr, sym.n_value, /*size=*/0,
        /*isWeakDef=*/false, sym.n_type & N_PEXT, sym.n_desc & N_ARM_THUMB_DEF,
        /*isReferencedDynamically=*/false, sym.n_desc & N_NO_DEAD_STRIP,
        /*isWeakDefCanBeHidden=*/false);
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

template <class NList> static bool isUndef(const NList &sym) {
  return (sym.n_type & N_TYPE) == N_UNDF && sym.n_value == 0;
}

template <class LP>
void ObjFile::parseSymbols(ArrayRef<typename LP::section> sectionHeaders,
                           ArrayRef<typename LP::nlist> nList,
                           const char *strtab, bool subsectionsViaSymbols) {
  using NList = typename LP::nlist;

  // Groups indices of the symbols by the sections that contain them.
  std::vector<std::vector<uint32_t>> symbolsBySection(sections.size());
  symbols.resize(nList.size());
  SmallVector<unsigned, 32> undefineds;
  for (uint32_t i = 0; i < nList.size(); ++i) {
    const NList &sym = nList[i];

    // Ignore debug symbols for now.
    // FIXME: may need special handling.
    if (sym.n_type & N_STAB)
      continue;

    StringRef name = strtab + sym.n_strx;
    if ((sym.n_type & N_TYPE) == N_SECT) {
      Subsections &subsections = sections[sym.n_sect - 1].subsections;
      // parseSections() may have chosen not to parse this section.
      if (subsections.empty())
        continue;
      symbolsBySection[sym.n_sect - 1].push_back(i);
    } else if (isUndef(sym)) {
      undefineds.push_back(i);
    } else {
      symbols[i] = parseNonSectionSymbol(sym, name);
    }
  }

  for (size_t i = 0; i < sections.size(); ++i) {
    Subsections &subsections = sections[i].subsections;
    if (subsections.empty())
      continue;
    InputSection *lastIsec = subsections.back().isec;
    if (lastIsec->getName() == section_names::ehFrame) {
      // __TEXT,__eh_frame only has symbols and SUBTRACTOR relocs when ld64 -r
      // adds local "EH_Frame1" and "func.eh". Ignore them because they have
      // gone unused by Mac OS since Snow Leopard (10.6), vintage 2009.
      continue;
    }
    std::vector<uint32_t> &symbolIndices = symbolsBySection[i];
    uint64_t sectionAddr = sectionHeaders[i].addr;
    uint32_t sectionAlign = 1u << sectionHeaders[i].align;

    // Record-based sections have already been split into subsections during
    // parseSections(), so we simply need to match Symbols to the corresponding
    // subsection here.
    if (getRecordSize(lastIsec->getSegName(), lastIsec->getName())) {
      for (size_t j = 0; j < symbolIndices.size(); ++j) {
        uint32_t symIndex = symbolIndices[j];
        const NList &sym = nList[symIndex];
        StringRef name = strtab + sym.n_strx;
        uint64_t symbolOffset = sym.n_value - sectionAddr;
        InputSection *isec =
            findContainingSubsection(subsections, &symbolOffset);
        if (symbolOffset != 0) {
          error(toString(lastIsec) + ":  symbol " + name +
                " at misaligned offset");
          continue;
        }
        symbols[symIndex] = createDefined(sym, name, isec, 0, isec->getSize());
      }
      continue;
    }

    // Calculate symbol sizes and create subsections by splitting the sections
    // along symbol boundaries.
    // We populate subsections by repeatedly splitting the last (highest
    // address) subsection.
    llvm::stable_sort(symbolIndices, [&](uint32_t lhs, uint32_t rhs) {
      return nList[lhs].n_value < nList[rhs].n_value;
    });
    for (size_t j = 0; j < symbolIndices.size(); ++j) {
      uint32_t symIndex = symbolIndices[j];
      const NList &sym = nList[symIndex];
      StringRef name = strtab + sym.n_strx;
      Subsection &subsec = subsections.back();
      InputSection *isec = subsec.isec;

      uint64_t subsecAddr = sectionAddr + subsec.offset;
      size_t symbolOffset = sym.n_value - subsecAddr;
      uint64_t symbolSize =
          j + 1 < symbolIndices.size()
              ? nList[symbolIndices[j + 1]].n_value - sym.n_value
              : isec->data.size() - symbolOffset;
      // There are 4 cases where we do not need to create a new subsection:
      //   1. If the input file does not use subsections-via-symbols.
      //   2. Multiple symbols at the same address only induce one subsection.
      //      (The symbolOffset == 0 check covers both this case as well as
      //      the first loop iteration.)
      //   3. Alternative entry points do not induce new subsections.
      //   4. If we have a literal section (e.g. __cstring and __literal4).
      if (!subsectionsViaSymbols || symbolOffset == 0 ||
          sym.n_desc & N_ALT_ENTRY || !isa<ConcatInputSection>(isec)) {
        symbols[symIndex] =
            createDefined(sym, name, isec, symbolOffset, symbolSize);
        continue;
      }
      auto *concatIsec = cast<ConcatInputSection>(isec);

      auto *nextIsec = make<ConcatInputSection>(*concatIsec);
      nextIsec->wasCoalesced = false;
      if (isZeroFill(isec->getFlags())) {
        // Zero-fill sections have NULL data.data() non-zero data.size()
        nextIsec->data = {nullptr, isec->data.size() - symbolOffset};
        isec->data = {nullptr, symbolOffset};
      } else {
        nextIsec->data = isec->data.slice(symbolOffset);
        isec->data = isec->data.slice(0, symbolOffset);
      }

      // By construction, the symbol will be at offset zero in the new
      // subsection.
      symbols[symIndex] =
          createDefined(sym, name, nextIsec, /*value=*/0, symbolSize);
      // TODO: ld64 appears to preserve the original alignment as well as each
      // subsection's offset from the last aligned address. We should consider
      // emulating that behavior.
      nextIsec->align = MinAlign(sectionAlign, sym.n_value);
      subsections.push_back({sym.n_value - sectionAddr, nextIsec});
    }
  }

  // Undefined symbols can trigger recursive fetch from Archives due to
  // LazySymbols. Process defined symbols first so that the relative order
  // between a defined symbol and an undefined symbol does not change the
  // symbol resolution behavior. In addition, a set of interconnected symbols
  // will all be resolved to the same file, instead of being resolved to
  // different files.
  for (unsigned i : undefineds) {
    const NList &sym = nList[i];
    StringRef name = strtab + sym.n_strx;
    symbols[i] = parseNonSectionSymbol(sym, name);
  }
}

OpaqueFile::OpaqueFile(MemoryBufferRef mb, StringRef segName,
                       StringRef sectName)
    : InputFile(OpaqueKind, mb) {
  const auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  ArrayRef<uint8_t> data = {buf, mb.getBufferSize()};
  ConcatInputSection *isec =
      make<ConcatInputSection>(segName.take_front(16), sectName.take_front(16),
                               /*file=*/this, data);
  isec->live = true;
  sections.push_back(0);
  sections.back().subsections.push_back({0, isec});
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
  using SectionHeader = typename LP::section;
  using NList = typename LP::nlist;

  auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  auto *hdr = reinterpret_cast<const Header *>(mb.getBufferStart());

  Architecture arch = getArchitectureFromCpuType(hdr->cputype, hdr->cpusubtype);
  if (arch != config->arch()) {
    auto msg = config->errorForArchMismatch
                   ? static_cast<void (*)(const Twine &)>(error)
                   : warn;
    msg(toString(this) + " has architecture " + getArchitectureName(arch) +
        " which is incompatible with target architecture " +
        getArchitectureName(config->arch()));
    return;
  }

  if (!checkCompatibility(this))
    return;

  for (auto *cmd : findCommands<linker_option_command>(hdr, LC_LINKER_OPTION)) {
    StringRef data{reinterpret_cast<const char *>(cmd + 1),
                   cmd->cmdsize - sizeof(linker_option_command)};
    parseLCLinkerOption(this, cmd->count, data);
  }

  ArrayRef<SectionHeader> sectionHeaders;
  if (const load_command *cmd = findCommand(hdr, LP::segmentLCType)) {
    auto *c = reinterpret_cast<const SegmentCommand *>(cmd);
    sectionHeaders = ArrayRef<SectionHeader>{
        reinterpret_cast<const SectionHeader *>(c + 1), c->nsects};
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
  for (size_t i = 0, n = sections.size(); i < n; ++i)
    if (!sections[i].subsections.empty())
      parseRelocations(sectionHeaders, sectionHeaders[i],
                       sections[i].subsections);

  parseDebugInfo();
  if (config->emitDataInCodeInfo)
    parseDataInCode();
  if (compactUnwindSection)
    registerCompactUnwind();
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

void ObjFile::parseDataInCode() {
  const auto *buf = reinterpret_cast<const uint8_t *>(mb.getBufferStart());
  const load_command *cmd = findCommand(buf, LC_DATA_IN_CODE);
  if (!cmd)
    return;
  const auto *c = reinterpret_cast<const linkedit_data_command *>(cmd);
  dataInCodeEntries = {
      reinterpret_cast<const data_in_code_entry *>(buf + c->dataoff),
      c->datasize / sizeof(data_in_code_entry)};
  assert(is_sorted(dataInCodeEntries, [](const data_in_code_entry &lhs,
                                         const data_in_code_entry &rhs) {
    return lhs.offset < rhs.offset;
  }));
}

// Create pointers from symbols to their associated compact unwind entries.
void ObjFile::registerCompactUnwind() {
  for (const Subsection &subsection : compactUnwindSection->subsections) {
    ConcatInputSection *isec = cast<ConcatInputSection>(subsection.isec);
    // Hack!! Since each CUE contains a different function address, if ICF
    // operated naively and compared the entire contents of each CUE, entries
    // with identical unwind info but belonging to different functions would
    // never be considered equivalent. To work around this problem, we slice
    // away the function address here. (Note that we do not adjust the offsets
    // of the corresponding relocations.) We rely on `relocateCompactUnwind()`
    // to correctly handle these truncated input sections.
    isec->data = isec->data.slice(target->wordSize);

    ConcatInputSection *referentIsec;
    for (auto it = isec->relocs.begin(); it != isec->relocs.end();) {
      Reloc &r = *it;
      // CUE::functionAddress is at offset 0. Skip personality & LSDA relocs.
      if (r.offset != 0) {
        ++it;
        continue;
      }
      uint64_t add = r.addend;
      if (auto *sym = cast_or_null<Defined>(r.referent.dyn_cast<Symbol *>())) {
        // Check whether the symbol defined in this file is the prevailing one.
        // Skip if it is e.g. a weak def that didn't prevail.
        if (sym->getFile() != this) {
          ++it;
          continue;
        }
        add += sym->value;
        referentIsec = cast<ConcatInputSection>(sym->isec);
      } else {
        referentIsec =
            cast<ConcatInputSection>(r.referent.dyn_cast<InputSection *>());
      }
      if (referentIsec->getSegName() != segment_names::text)
        error("compact unwind references address in " + toString(referentIsec) +
              " which is not in segment __TEXT");
      // The functionAddress relocations are typically section relocations.
      // However, unwind info operates on a per-symbol basis, so we search for
      // the function symbol here.
      auto symIt = llvm::lower_bound(
          referentIsec->symbols, add,
          [](Defined *d, uint64_t add) { return d->value < add; });
      // The relocation should point at the exact address of a symbol (with no
      // addend).
      if (symIt == referentIsec->symbols.end() || (*symIt)->value != add) {
        assert(referentIsec->wasCoalesced);
        ++it;
        continue;
      }
      (*symIt)->unwindEntry = isec;
      // Since we've sliced away the functionAddress, we should remove the
      // corresponding relocation too. Given that clang emits relocations in
      // reverse order of address, this relocation should be at the end of the
      // vector for most of our input object files, so this is typically an O(1)
      // operation.
      it = isec->relocs.erase(it);
    }
  }
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
static DylibFile *findDylib(StringRef path, DylibFile *umbrella,
                            const InterfaceFile *currentTopLevelTapi) {
  // Search order:
  // 1. Install name basename in -F / -L directories.
  {
    StringRef stem = path::stem(path);
    SmallString<128> frameworkName;
    path::append(frameworkName, path::Style::posix, stem + ".framework", stem);
    bool isFramework = path.endswith(frameworkName);
    if (isFramework) {
      for (StringRef dir : config->frameworkSearchPaths) {
        SmallString<128> candidate = dir;
        path::append(candidate, frameworkName);
        if (Optional<StringRef> dylibPath = resolveDylibPath(candidate.str()))
          return loadDylib(*dylibPath, umbrella);
      }
    } else if (Optional<StringRef> dylibPath = findPathCombination(
                   stem, config->librarySearchPaths, {".tbd", ".dylib"}))
      return loadDylib(*dylibPath, umbrella);
  }

  // 2. As absolute path.
  if (path::is_absolute(path, path::Style::posix))
    for (StringRef root : config->systemLibraryRoots)
      if (Optional<StringRef> dylibPath = resolveDylibPath((root + path).str()))
        return loadDylib(*dylibPath, umbrella);

  // 3. As relative path.

  // TODO: Handle -dylib_file

  // Replace @executable_path, @loader_path, @rpath prefixes in install name.
  SmallString<128> newPath;
  if (config->outputType == MH_EXECUTE &&
      path.consume_front("@executable_path/")) {
    // ld64 allows overriding this with the undocumented flag -executable_path.
    // lld doesn't currently implement that flag.
    // FIXME: Consider using finalOutput instead of outputFile.
    path::append(newPath, path::parent_path(config->outputFile), path);
    path = newPath;
  } else if (path.consume_front("@loader_path/")) {
    fs::real_path(umbrella->getName(), newPath);
    path::remove_filename(newPath);
    path::append(newPath, path);
    path = newPath;
  } else if (path.startswith("@rpath/")) {
    for (StringRef rpath : umbrella->rpaths) {
      newPath.clear();
      if (rpath.consume_front("@loader_path/")) {
        fs::real_path(umbrella->getName(), newPath);
        path::remove_filename(newPath);
      }
      path::append(newPath, rpath, path.drop_front(strlen("@rpath/")));
      if (Optional<StringRef> dylibPath = resolveDylibPath(newPath.str()))
        return loadDylib(*dylibPath, umbrella);
    }
  }

  // FIXME: Should this be further up?
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

  if (Optional<StringRef> dylibPath = resolveDylibPath(path))
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

static void loadReexport(StringRef path, DylibFile *umbrella,
                         const InterfaceFile *currentTopLevelTapi) {
  DylibFile *reexport = findDylib(path, umbrella, currentTopLevelTapi);
  if (!reexport)
    error("unable to locate re-export with install name " + path);
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

  // Initialize installName.
  if (const load_command *cmd = findCommand(hdr, LC_ID_DYLIB)) {
    auto *c = reinterpret_cast<const dylib_command *>(cmd);
    currentVersion = read32le(&c->dylib.current_version);
    compatibilityVersion = read32le(&c->dylib.compatibility_version);
    installName =
        reinterpret_cast<const char *>(cmd) + read32le(&c->dylib.name);
  } else if (!isBundleLoader) {
    // macho_executable and macho_bundle don't have LC_ID_DYLIB,
    // so it's OK.
    error("dylib " + toString(this) + " missing LC_ID_DYLIB load command");
    return;
  }

  if (config->printEachFile)
    message(toString(this));
  inputFiles.insert(this);

  deadStrippable = hdr->flags & MH_DEAD_STRIPPABLE_DYLIB;

  if (!checkCompatibility(this))
    return;

  checkAppExtensionSafety(hdr->flags & MH_APP_EXTENSION_SAFE);

  for (auto *cmd : findCommands<rpath_command>(hdr, LC_RPATH)) {
    StringRef rpath{reinterpret_cast<const char *>(cmd) + cmd->path};
    rpaths.push_back(rpath);
  }

  // Initialize symbols.
  exportingFile = isImplicitlyLinked(installName) ? this : this->umbrella;
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

  installName = saver.save(interface.getInstallName());
  compatibilityVersion = interface.getCompatibilityVersion().rawValue();
  currentVersion = interface.getCurrentVersion().rawValue();

  if (config->printEachFile)
    message(toString(this));
  inputFiles.insert(this);

  if (!is_contained(skipPlatformChecks, installName) &&
      !is_contained(interface.targets(), config->platformInfo.target)) {
    error(toString(this) + " is incompatible with " +
          std::string(config->platformInfo.target));
    return;
  }

  checkAppExtensionSafety(interface.isApplicationExtensionSafe());

  exportingFile = isImplicitlyLinked(installName) ? this : umbrella;
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
  for (const InterfaceFileRef &intfRef : interface.reexportedLibraries()) {
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

  this->installName = saver.save(installName);

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
    this->installName = saver.save(installName);
}

void DylibFile::checkAppExtensionSafety(bool dylibIsAppExtensionSafe) const {
  if (config->applicationExtension && !dylibIsAppExtensionSafe)
    warn("using '-application_extension' with unsafe dylib: " + toString(this));
}

ArchiveFile::ArchiveFile(std::unique_ptr<object::Archive> &&f)
    : InputFile(ArchiveKind, f->getMemoryBufferRef()), file(std::move(f)) {}

void ArchiveFile::addLazySymbols() {
  for (const object::Archive::Symbol &sym : file->symbols())
    symtab->addLazy(sym.getName(), this, sym);
}

static Expected<InputFile *> loadArchiveMember(MemoryBufferRef mb,
                                               uint32_t modTime,
                                               StringRef archiveName,
                                               uint64_t offsetInArchive) {
  if (config->zeroModTime)
    modTime = 0;

  switch (identify_magic(mb.getBuffer())) {
  case file_magic::macho_object:
    return make<ObjFile>(mb, modTime, archiveName);
  case file_magic::bitcode:
    return make<BitcodeFile>(mb, archiveName, offsetInArchive);
  default:
    return createStringError(inconvertibleErrorCode(),
                             mb.getBufferIdentifier() +
                                 " has unhandled file type");
  }
}

Error ArchiveFile::fetch(const object::Archive::Child &c, StringRef reason) {
  if (!seen.insert(c.getChildOffset()).second)
    return Error::success();

  Expected<MemoryBufferRef> mb = c.getMemoryBufferRef();
  if (!mb)
    return mb.takeError();

  // Thin archives refer to .o files, so --reproduce needs the .o files too.
  if (tar && c.getParent()->isThin())
    tar->append(relativeToRoot(CHECK(c.getFullName(), this)), mb->getBuffer());

  Expected<TimePoint<std::chrono::seconds>> modTime = c.getLastModified();
  if (!modTime)
    return modTime.takeError();

  Expected<InputFile *> file =
      loadArchiveMember(*mb, toTimeT(*modTime), getName(), c.getChildOffset());

  if (!file)
    return file.takeError();

  inputFiles.insert(*file);
  printArchiveMemberLoad(reason, *file);
  return Error::success();
}

void ArchiveFile::fetch(const object::Archive::Symbol &sym) {
  object::Archive::Child c =
      CHECK(sym.getMember(), toString(this) +
                                 ": could not get the member defining symbol " +
                                 toMachOString(sym));

  // `sym` is owned by a LazySym, which will be replace<>()d by make<ObjFile>
  // and become invalid after that call. Copy it to the stack so we can refer
  // to it later.
  const object::Archive::Symbol symCopy = sym;

  // ld64 doesn't demangle sym here even with -demangle.
  // Match that: intentionally don't call toMachOString().
  if (Error e = fetch(c, symCopy.getName()))
    error(toString(this) + ": could not get the member defining symbol " +
          toMachOString(symCopy) + ": " + toString(std::move(e)));
}

static macho::Symbol *createBitcodeSymbol(const lto::InputFile::Symbol &objSym,
                                          BitcodeFile &file) {
  StringRef name = saver.save(objSym.getName());

  // TODO: support weak references
  if (objSym.isUndefined())
    return symtab->addUndefined(name, &file, /*isWeakRef=*/false);

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

  if (objSym.isCommon())
    return symtab->addCommon(name, &file, objSym.getCommonSize(),
                             objSym.getCommonAlignment(), isPrivateExtern);

  return symtab->addDefined(name, &file, /*isec=*/nullptr, /*value=*/0,
                            /*size=*/0, objSym.isWeak(), isPrivateExtern,
                            /*isThumb=*/false,
                            /*isReferencedDynamically=*/false,
                            /*noDeadStrip=*/false,
                            /*isWeakDefCanBeHidden=*/false);
}

BitcodeFile::BitcodeFile(MemoryBufferRef mb, StringRef archiveName,
                         uint64_t offsetInArchive)
    : InputFile(BitcodeKind, mb) {
  std::string path = mb.getBufferIdentifier().str();
  // ThinLTO assumes that all MemoryBufferRefs given to it have a unique
  // name. If two members with the same name are provided, this causes a
  // collision and ThinLTO can't proceed.
  // So, we append the archive name to disambiguate two members with the same
  // name from multiple different archives, and offset within the archive to
  // disambiguate two members of the same name from a single archive.
  MemoryBufferRef mbref(
      mb.getBuffer(),
      saver.save(archiveName.empty() ? path
                                     : archiveName + sys::path::filename(path) +
                                           utostr(offsetInArchive)));

  obj = check(lto::InputFile::create(mbref));

  // Convert LTO Symbols to LLD Symbols in order to perform resolution. The
  // "winning" symbol will then be marked as Prevailing at LTO compilation
  // time.
  for (const lto::InputFile::Symbol &objSym : obj->symbols())
    symbols.push_back(createBitcodeSymbol(objSym, *this));
}

template void ObjFile::parse<LP64>();
