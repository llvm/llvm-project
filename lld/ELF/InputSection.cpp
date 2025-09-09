//===- InputSection.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Config.h"
#include "InputFiles.h"
#include "OutputSections.h"
#include "Relocations.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/DWARF.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <optional>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace llvm::sys;
using namespace lld;
using namespace lld::elf;

// Returns a string to construct an error message.
std::string elf::toStr(Ctx &ctx, const InputSectionBase *sec) {
  return (toStr(ctx, sec->file) + ":(" + sec->name + ")").str();
}

const ELFSyncStream &elf::operator<<(const ELFSyncStream &s,
                                     const InputSectionBase *sec) {
  return s << toStr(s.ctx, sec);
}

template <class ELFT>
static ArrayRef<uint8_t> getSectionContents(ObjFile<ELFT> &file,
                                            const typename ELFT::Shdr &hdr) {
  if (hdr.sh_type == SHT_NOBITS)
    return ArrayRef<uint8_t>(nullptr, hdr.sh_size);
  return check(file.getObj().getSectionContents(hdr));
}

InputSectionBase::InputSectionBase(InputFile *file, StringRef name,
                                   uint32_t type, uint64_t flags, uint32_t link,
                                   uint32_t info, uint32_t addralign,
                                   uint32_t entsize, ArrayRef<uint8_t> data,
                                   Kind sectionKind)
    : SectionBase(sectionKind, file, name, type, flags, link, info, addralign,
                  entsize),
      bss(0), decodedCrel(0), keepUnique(0), nopFiller(0),
      content_(data.data()), size(data.size()) {
  // In order to reduce memory allocation, we assume that mergeable
  // sections are smaller than 4 GiB, which is not an unreasonable
  // assumption as of 2017.
  if (sectionKind == SectionBase::Merge && content().size() > UINT32_MAX)
    ErrAlways(getCtx()) << this << ": section too large";

  // The ELF spec states that a value of 0 means the section has
  // no alignment constraints.
  uint32_t v = std::max<uint32_t>(addralign, 1);
  if (!isPowerOf2_64(v)) {
    Err(getCtx()) << this << ": sh_addralign is not a power of 2";
    v = 1;
  }
  this->addralign = v;

  // If SHF_COMPRESSED is set, parse the header. The legacy .zdebug format is no
  // longer supported.
  if (flags & SHF_COMPRESSED) {
    Ctx &ctx = file->ctx;
    invokeELFT(parseCompressedHeader, ctx);
  }
}

// SHF_INFO_LINK and SHF_GROUP are normally resolved and not copied to the
// output section. However, for relocatable linking without
// --force-group-allocation, the SHF_GROUP flag and section groups are retained.
static uint64_t getFlags(Ctx &ctx, uint64_t flags) {
  flags &= ~(uint64_t)SHF_INFO_LINK;
  if (ctx.arg.resolveGroups)
    flags &= ~(uint64_t)SHF_GROUP;
  return flags;
}

template <class ELFT>
InputSectionBase::InputSectionBase(ObjFile<ELFT> &file,
                                   const typename ELFT::Shdr &hdr,
                                   StringRef name, Kind sectionKind)
    : InputSectionBase(&file, name, hdr.sh_type,
                       getFlags(file.ctx, hdr.sh_flags), hdr.sh_link,
                       hdr.sh_info, hdr.sh_addralign, hdr.sh_entsize,
                       getSectionContents(file, hdr), sectionKind) {
  // We reject object files having insanely large alignments even though
  // they are allowed by the spec. I think 4GB is a reasonable limitation.
  // We might want to relax this in the future.
  if (hdr.sh_addralign > UINT32_MAX) {
    Err(getCtx()) << &file << ": section sh_addralign is too large";
    addralign = 1;
  }
}

size_t InputSectionBase::getSize() const {
  if (auto *s = dyn_cast<SyntheticSection>(this))
    return s->getSize();
  return size - bytesDropped;
}

template <class ELFT>
static void decompressAux(Ctx &ctx, const InputSectionBase &sec, uint8_t *out,
                          size_t size) {
  auto *hdr = reinterpret_cast<const typename ELFT::Chdr *>(sec.content_);
  auto compressed = ArrayRef<uint8_t>(sec.content_, sec.compressedSize)
                        .slice(sizeof(typename ELFT::Chdr));
  if (Error e = hdr->ch_type == ELFCOMPRESS_ZLIB
                    ? compression::zlib::decompress(compressed, out, size)
                    : compression::zstd::decompress(compressed, out, size))
    Err(ctx) << &sec << ": decompress failed: " << std::move(e);
}

void InputSectionBase::decompress() const {
  Ctx &ctx = getCtx();
  uint8_t *buf = makeThreadLocalN<uint8_t>(size);
  invokeELFT(decompressAux, ctx, *this, buf, size);
  content_ = buf;
  compressed = false;
}

template <class ELFT>
RelsOrRelas<ELFT> InputSectionBase::relsOrRelas(bool supportsCrel) const {
  if (relSecIdx == 0)
    return {};
  RelsOrRelas<ELFT> ret;
  auto *f = cast<ObjFile<ELFT>>(file);
  typename ELFT::Shdr shdr = f->template getELFShdrs<ELFT>()[relSecIdx];
  if (shdr.sh_type == SHT_CREL) {
    // Return an iterator if supported by caller.
    if (supportsCrel) {
      ret.crels = Relocs<typename ELFT::Crel>(
          (const uint8_t *)f->mb.getBufferStart() + shdr.sh_offset);
      return ret;
    }
    InputSectionBase *const &relSec = f->getSections()[relSecIdx];
    // Otherwise, allocate a buffer to hold the decoded RELA relocations. When
    // called for the first time, relSec is null (without --emit-relocs) or an
    // InputSection with false decodedCrel.
    if (!relSec || !cast<InputSection>(relSec)->decodedCrel) {
      auto *sec = makeThreadLocal<InputSection>(*f, shdr, name);
      f->cacheDecodedCrel(relSecIdx, sec);
      sec->type = SHT_RELA;
      sec->decodedCrel = true;

      RelocsCrel<ELFT::Is64Bits> entries(sec->content_);
      sec->size = entries.size() * sizeof(typename ELFT::Rela);
      auto *relas = makeThreadLocalN<typename ELFT::Rela>(entries.size());
      sec->content_ = reinterpret_cast<uint8_t *>(relas);
      for (auto [i, r] : llvm::enumerate(entries)) {
        relas[i].r_offset = r.r_offset;
        relas[i].setSymbolAndType(r.r_symidx, r.r_type, false);
        relas[i].r_addend = r.r_addend;
      }
    }
    ret.relas = {ArrayRef(
        reinterpret_cast<const typename ELFT::Rela *>(relSec->content_),
        relSec->size / sizeof(typename ELFT::Rela))};
    return ret;
  }

  const void *content = f->mb.getBufferStart() + shdr.sh_offset;
  size_t size = shdr.sh_size;
  if (shdr.sh_type == SHT_REL) {
    ret.rels = {ArrayRef(reinterpret_cast<const typename ELFT::Rel *>(content),
                         size / sizeof(typename ELFT::Rel))};
  } else {
    assert(shdr.sh_type == SHT_RELA);
    ret.relas = {
        ArrayRef(reinterpret_cast<const typename ELFT::Rela *>(content),
                 size / sizeof(typename ELFT::Rela))};
  }
  return ret;
}

Ctx &SectionBase::getCtx() const { return file->ctx; }

uint64_t SectionBase::getOffset(uint64_t offset) const {
  switch (kind()) {
  case Output: {
    auto *os = cast<OutputSection>(this);
    // For output sections we treat offset -1 as the end of the section.
    return offset == uint64_t(-1) ? os->size : offset;
  }
  case Class:
    llvm_unreachable("section classes do not have offsets");
  case Regular:
  case Synthetic:
  case Spill:
    return cast<InputSection>(this)->outSecOff + offset;
  case EHFrame: {
    // Two code paths may reach here. First, clang_rt.crtbegin.o and GCC
    // crtbeginT.o may reference the start of an empty .eh_frame to identify the
    // start of the output .eh_frame. Just return offset.
    //
    // Second, InputSection::copyRelocations on .eh_frame. Some pieces may be
    // discarded due to GC/ICF. We should compute the output section offset.
    const EhInputSection *es = cast<EhInputSection>(this);
    if (!es->content().empty())
      if (InputSection *isec = es->getParent())
        return isec->outSecOff + es->getParentOffset(offset);
    return offset;
  }
  case Merge:
    const MergeInputSection *ms = cast<MergeInputSection>(this);
    if (InputSection *isec = ms->getParent())
      return isec->outSecOff + ms->getParentOffset(offset);
    return ms->getParentOffset(offset);
  }
  llvm_unreachable("invalid section kind");
}

uint64_t SectionBase::getVA(uint64_t offset) const {
  const OutputSection *out = getOutputSection();
  return (out ? out->addr : 0) + getOffset(offset);
}

OutputSection *SectionBase::getOutputSection() {
  InputSection *sec;
  if (auto *isec = dyn_cast<InputSection>(this))
    sec = isec;
  else if (auto *ms = dyn_cast<MergeInputSection>(this))
    sec = ms->getParent();
  else if (auto *eh = dyn_cast<EhInputSection>(this))
    sec = eh->getParent();
  else
    return cast<OutputSection>(this);
  return sec ? sec->getParent() : nullptr;
}

// When a section is compressed, `rawData` consists with a header followed
// by zlib-compressed data. This function parses a header to initialize
// `uncompressedSize` member and remove the header from `rawData`.
template <typename ELFT>
void InputSectionBase::parseCompressedHeader(Ctx &ctx) {
  flags &= ~(uint64_t)SHF_COMPRESSED;

  // New-style header
  if (content().size() < sizeof(typename ELFT::Chdr)) {
    ErrAlways(ctx) << this << ": corrupted compressed section";
    return;
  }

  auto *hdr = reinterpret_cast<const typename ELFT::Chdr *>(content().data());
  if (hdr->ch_type == ELFCOMPRESS_ZLIB) {
    if (!compression::zlib::isAvailable())
      ErrAlways(ctx) << this
                     << " is compressed with ELFCOMPRESS_ZLIB, but lld is "
                        "not built with zlib support";
  } else if (hdr->ch_type == ELFCOMPRESS_ZSTD) {
    if (!compression::zstd::isAvailable())
      ErrAlways(ctx) << this
                     << " is compressed with ELFCOMPRESS_ZSTD, but lld is "
                        "not built with zstd support";
  } else {
    ErrAlways(ctx) << this << ": unsupported compression type ("
                   << uint32_t(hdr->ch_type) << ")";
    return;
  }

  compressed = true;
  compressedSize = size;
  size = hdr->ch_size;
  addralign = std::max<uint32_t>(hdr->ch_addralign, 1);
}

InputSection *InputSectionBase::getLinkOrderDep() const {
  assert(flags & SHF_LINK_ORDER);
  if (!link)
    return nullptr;
  return cast<InputSection>(file->getSections()[link]);
}

// Find a symbol that encloses a given location.
Defined *InputSectionBase::getEnclosingSymbol(uint64_t offset,
                                              uint8_t type) const {
  if (file->isInternal())
    return nullptr;
  for (Symbol *b : file->getSymbols())
    if (Defined *d = dyn_cast<Defined>(b))
      if (d->section == this && d->value <= offset &&
          offset < d->value + d->size && (type == 0 || type == d->type))
        return d;
  return nullptr;
}

// Returns an object file location string. Used to construct an error message.
std::string InputSectionBase::getLocation(uint64_t offset) const {
  std::string secAndOffset =
      (name + "+0x" + Twine::utohexstr(offset) + ")").str();

  std::string filename = toStr(getCtx(), file);
  if (Defined *d = getEnclosingFunction(offset))
    return filename + ":(function " + toStr(getCtx(), *d) + ": " + secAndOffset;

  return filename + ":(" + secAndOffset;
}

static void printFileLine(const ELFSyncStream &s, StringRef path,
                          unsigned line) {
  StringRef filename = path::filename(path);
  s << filename << ':' << line;
  if (filename != path)
    s << " (" << path << ':' << line << ')';
}

// Print an error message that looks like this:
//
//   foo.c:42 (/home/alice/possibly/very/long/path/foo.c:42)
const ELFSyncStream &elf::operator<<(const ELFSyncStream &s,
                                     InputSectionBase::SrcMsg &&msg) {
  auto &sec = msg.sec;
  if (sec.file->kind() != InputFile::ObjKind)
    return s;
  auto &file = cast<ELFFileBase>(*sec.file);

  // First, look up the DWARF line table.
  ArrayRef<InputSectionBase *> sections = file.getSections();
  auto it = llvm::find(sections, &sec);
  uint64_t sectionIndex = it != sections.end()
                              ? it - sections.begin()
                              : object::SectionedAddress::UndefSection;
  DWARFCache *dwarf = file.getDwarf();
  if (auto info = dwarf->getDILineInfo(msg.offset, sectionIndex))
    printFileLine(s, info->FileName, info->Line);
  else if (auto fileLine = dwarf->getVariableLoc(msg.sym.getName()))
    // If it failed, look up again as a variable.
    printFileLine(s, fileLine->first, fileLine->second);
  else
    // File.sourceFile contains STT_FILE symbol, and that is a last resort.
    s << file.sourceFile;
  return s;
}

// Returns a filename string along with an optional section name. This
// function is intended to be used for constructing an error
// message. The returned message looks like this:
//
//   path/to/foo.o:(function bar)
//
// or
//
//   path/to/foo.o:(function bar) in archive path/to/bar.a
const ELFSyncStream &elf::operator<<(const ELFSyncStream &s,
                                     InputSectionBase::ObjMsg &&msg) {
  auto *sec = msg.sec;
  s << sec->file->getName() << ":(";

  // Find a symbol that encloses a given location. getObjMsg may be called
  // before ObjFile::initSectionsAndLocalSyms where local symbols are
  // initialized.
  if (Defined *d = sec->getEnclosingSymbol(msg.offset))
    s << d;
  else
    s << sec->name << "+0x" << Twine::utohexstr(msg.offset);
  s << ')';
  if (!sec->file->archiveName.empty())
    s << (" in archive " + sec->file->archiveName).str();
  return s;
}

PotentialSpillSection::PotentialSpillSection(const InputSectionBase &source,
                                             InputSectionDescription &isd)
    : InputSection(source.file, source.name, source.type, source.flags,
                   source.addralign, source.addralign, {}, SectionBase::Spill),
      isd(&isd) {}

InputSection InputSection::discarded(nullptr, "", 0, 0, 0, 0,
                                     ArrayRef<uint8_t>());

InputSection::InputSection(InputFile *f, StringRef name, uint32_t type,
                           uint64_t flags, uint32_t addralign, uint32_t entsize,
                           ArrayRef<uint8_t> data, Kind k)
    : InputSectionBase(f, name, type, flags,
                       /*link=*/0, /*info=*/0, addralign, /*entsize=*/entsize,
                       data, k) {
  assert(f || this == &InputSection::discarded);
}

template <class ELFT>
InputSection::InputSection(ObjFile<ELFT> &f, const typename ELFT::Shdr &header,
                           StringRef name)
    : InputSectionBase(f, header, name, InputSectionBase::Regular) {}

// Copy SHT_GROUP section contents. Used only for the -r option.
template <class ELFT> void InputSection::copyShtGroup(uint8_t *buf) {
  // ELFT::Word is the 32-bit integral type in the target endianness.
  using u32 = typename ELFT::Word;
  ArrayRef<u32> from = getDataAs<u32>();
  auto *to = reinterpret_cast<u32 *>(buf);

  // The first entry is not a section number but a flag.
  *to++ = from[0];

  // Adjust section numbers because section numbers in an input object files are
  // different in the output. We also need to handle combined or discarded
  // members.
  ArrayRef<InputSectionBase *> sections = file->getSections();
  DenseSet<uint32_t> seen;
  for (uint32_t idx : from.slice(1)) {
    OutputSection *osec = sections[idx]->getOutputSection();
    if (osec && seen.insert(osec->sectionIndex).second)
      *to++ = osec->sectionIndex;
  }
}

InputSectionBase *InputSection::getRelocatedSection() const {
  if (file->isInternal() || !isStaticRelSecType(type))
    return nullptr;
  ArrayRef<InputSectionBase *> sections = file->getSections();
  return sections[info];
}

template <class ELFT, class RelTy>
void InputSection::copyRelocations(Ctx &ctx, uint8_t *buf) {
  bool linkerRelax =
      ctx.arg.relax && is_contained({EM_RISCV, EM_LOONGARCH}, ctx.arg.emachine);
  if (!ctx.arg.relocatable && (linkerRelax || ctx.arg.branchToBranch)) {
    // On LoongArch and RISC-V, relaxation might change relocations: copy
    // from internal ones that are updated by relaxation.
    InputSectionBase *sec = getRelocatedSection();
    copyRelocations<ELFT, RelTy>(
        ctx, buf,
        llvm::make_range(sec->relocations.begin(), sec->relocations.end()));
  } else {
    // Convert the raw relocations in the input section into Relocation objects
    // suitable to be used by copyRelocations below.
    struct MapRel {
      Ctx &ctx;
      const ObjFile<ELFT> &file;
      Relocation operator()(const RelTy &rel) const {
        // RelExpr is not used so set to a dummy value.
        return Relocation{R_NONE, rel.getType(ctx.arg.isMips64EL), rel.r_offset,
                          getAddend<ELFT>(rel), &file.getRelocTargetSym(rel)};
      }
    };

    using RawRels = ArrayRef<RelTy>;
    using MapRelIter =
        llvm::mapped_iterator<typename RawRels::iterator, MapRel>;
    auto mapRel = MapRel{ctx, *getFile<ELFT>()};
    RawRels rawRels = getDataAs<RelTy>();
    auto rels = llvm::make_range(MapRelIter(rawRels.begin(), mapRel),
                                 MapRelIter(rawRels.end(), mapRel));
    copyRelocations<ELFT, RelTy>(ctx, buf, rels);
  }
}

// This is used for -r and --emit-relocs. We can't use memcpy to copy
// relocations because we need to update symbol table offset and section index
// for each relocation. So we copy relocations one by one.
template <class ELFT, class RelTy, class RelIt>
void InputSection::copyRelocations(Ctx &ctx, uint8_t *buf,
                                   llvm::iterator_range<RelIt> rels) {
  const TargetInfo &target = *ctx.target;
  InputSectionBase *sec = getRelocatedSection();
  (void)sec->contentMaybeDecompress(); // uncompress if needed

  for (const Relocation &rel : rels) {
    RelType type = rel.type;
    const ObjFile<ELFT> *file = getFile<ELFT>();
    Symbol &sym = *rel.sym;

    auto *p = reinterpret_cast<typename ELFT::Rela *>(buf);
    buf += sizeof(RelTy);

    if (RelTy::HasAddend)
      p->r_addend = rel.addend;

    // Output section VA is zero for -r, so r_offset is an offset within the
    // section, but for --emit-relocs it is a virtual address.
    p->r_offset = sec->getVA(rel.offset);
    p->setSymbolAndType(ctx.in.symTab->getSymbolIndex(sym), type,
                        ctx.arg.isMips64EL);

    if (sym.type == STT_SECTION) {
      // We combine multiple section symbols into only one per
      // section. This means we have to update the addend. That is
      // trivial for Elf_Rela, but for Elf_Rel we have to write to the
      // section data. We do that by adding to the Relocation vector.

      // .eh_frame is horribly special and can reference discarded sections. To
      // avoid having to parse and recreate .eh_frame, we just replace any
      // relocation in it pointing to discarded sections with R_*_NONE, which
      // hopefully creates a frame that is ignored at runtime. Also, don't warn
      // on .gcc_except_table and debug sections.
      //
      // See the comment in maybeReportUndefined for PPC32 .got2 and PPC64 .toc
      auto *d = dyn_cast<Defined>(&sym);
      if (!d) {
        if (!isDebugSection(*sec) && sec->name != ".eh_frame" &&
            sec->name != ".gcc_except_table" && sec->name != ".got2" &&
            sec->name != ".toc") {
          uint32_t secIdx = cast<Undefined>(sym).discardedSecIdx;
          Elf_Shdr_Impl<ELFT> sec = file->template getELFShdrs<ELFT>()[secIdx];
          Warn(ctx) << "relocation refers to a discarded section: "
                    << CHECK2(file->getObj().getSectionName(sec), file)
                    << "\n>>> referenced by " << getObjMsg(p->r_offset);
        }
        p->setSymbolAndType(0, 0, false);
        continue;
      }
      SectionBase *section = d->section;
      assert(section->isLive());

      int64_t addend = rel.addend;
      const uint8_t *bufLoc = sec->content().begin() + rel.offset;
      if (!RelTy::HasAddend)
        addend = target.getImplicitAddend(bufLoc, type);

      if (ctx.arg.emachine == EM_MIPS &&
          target.getRelExpr(type, sym, bufLoc) == RE_MIPS_GOTREL) {
        // Some MIPS relocations depend on "gp" value. By default,
        // this value has 0x7ff0 offset from a .got section. But
        // relocatable files produced by a compiler or a linker
        // might redefine this default value and we must use it
        // for a calculation of the relocation result. When we
        // generate EXE or DSO it's trivial. Generating a relocatable
        // output is more difficult case because the linker does
        // not calculate relocations in this mode and loses
        // individual "gp" values used by each input object file.
        // As a workaround we add the "gp" value to the relocation
        // addend and save it back to the file.
        addend += sec->getFile<ELFT>()->mipsGp0;
      }

      if (RelTy::HasAddend)
        p->r_addend =
            sym.getVA(ctx, addend) - section->getOutputSection()->addr;
      // For SHF_ALLOC sections relocated by REL, append a relocation to
      // sec->relocations so that relocateAlloc transitively called by
      // writeSections will update the implicit addend. Non-SHF_ALLOC sections
      // utilize relocateNonAlloc to process raw relocations and do not need
      // this sec->relocations change.
      else if (ctx.arg.relocatable && (sec->flags & SHF_ALLOC) &&
               type != target.noneRel)
        sec->addReloc({R_ABS, type, rel.offset, addend, &sym});
    } else if (ctx.arg.emachine == EM_PPC && type == R_PPC_PLTREL24 &&
               p->r_addend >= 0x8000 && sec->file->ppc32Got2) {
      // Similar to R_MIPS_GPREL{16,32}. If the addend of R_PPC_PLTREL24
      // indicates that r30 is relative to the input section .got2
      // (r_addend>=0x8000), after linking, r30 should be relative to the output
      // section .got2 . To compensate for the shift, adjust r_addend by
      // ppc32Got->outSecOff.
      p->r_addend += sec->file->ppc32Got2->outSecOff;
    }
  }
}

// The ARM and AArch64 ABI handle pc-relative relocations to undefined weak
// references specially. The general rule is that the value of the symbol in
// this context is the address of the place P. A further special case is that
// branch relocations to an undefined weak reference resolve to the next
// instruction.
static uint32_t getARMUndefinedRelativeWeakVA(RelType type, uint32_t a,
                                              uint32_t p) {
  switch (type) {
  // Unresolved branch relocations to weak references resolve to next
  // instruction, this will be either 2 or 4 bytes on from P.
  case R_ARM_THM_JUMP8:
  case R_ARM_THM_JUMP11:
    return p + 2 + a;
  case R_ARM_CALL:
  case R_ARM_JUMP24:
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_PREL31:
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
    return p + 4 + a;
  case R_ARM_THM_CALL:
    // We don't want an interworking BLX to ARM
    return p + 5 + a;
  // Unresolved non branch pc-relative relocations
  // R_ARM_TARGET2 which can be resolved relatively is not present as it never
  // targets a weak-reference.
  case R_ARM_MOVW_PREL_NC:
  case R_ARM_MOVT_PREL:
  case R_ARM_REL32:
  case R_ARM_THM_ALU_PREL_11_0:
  case R_ARM_THM_MOVW_PREL_NC:
  case R_ARM_THM_MOVT_PREL:
  case R_ARM_THM_PC12:
    return p + a;
  // p + a is unrepresentable as negative immediates can't be encoded.
  case R_ARM_THM_PC8:
    return p;
  }
  llvm_unreachable("ARM pc-relative relocation expected\n");
}

// The comment above getARMUndefinedRelativeWeakVA applies to this function.
static uint64_t getAArch64UndefinedRelativeWeakVA(uint64_t type, uint64_t p) {
  switch (type) {
  // Unresolved branch relocations to weak references resolve to next
  // instruction, this is 4 bytes on from P.
  case R_AARCH64_CALL26:
  case R_AARCH64_CONDBR19:
  case R_AARCH64_JUMP26:
  case R_AARCH64_TSTBR14:
    return p + 4;
  // Unresolved non branch pc-relative relocations
  case R_AARCH64_PREL16:
  case R_AARCH64_PREL32:
  case R_AARCH64_PREL64:
  case R_AARCH64_ADR_PREL_LO21:
  case R_AARCH64_LD_PREL_LO19:
  case R_AARCH64_PLT32:
    return p;
  }
  llvm_unreachable("AArch64 pc-relative relocation expected\n");
}

static uint64_t getRISCVUndefinedRelativeWeakVA(uint64_t type, uint64_t p) {
  switch (type) {
  case R_RISCV_BRANCH:
  case R_RISCV_JAL:
  case R_RISCV_CALL:
  case R_RISCV_CALL_PLT:
  case R_RISCV_RVC_BRANCH:
  case R_RISCV_RVC_JUMP:
  case R_RISCV_PLT32:
    return p;
  default:
    return 0;
  }
}

// ARM SBREL relocations are of the form S + A - B where B is the static base
// The ARM ABI defines base to be "addressing origin of the output segment
// defining the symbol S". We defined the "addressing origin"/static base to be
// the base of the PT_LOAD segment containing the Sym.
// The procedure call standard only defines a Read Write Position Independent
// RWPI variant so in practice we should expect the static base to be the base
// of the RW segment.
static uint64_t getARMStaticBase(const Symbol &sym) {
  OutputSection *os = sym.getOutputSection();
  if (!os || !os->ptLoad || !os->ptLoad->firstSec) {
    Err(os->ctx) << "SBREL relocation to " << sym.getName()
                 << " without static base";
    return 0;
  }
  return os->ptLoad->firstSec->addr;
}

// For RE_RISCV_PC_INDIRECT (R_RISCV_PCREL_LO12_{I,S}), the symbol actually
// points the corresponding R_RISCV_PCREL_HI20 relocation, and the target VA
// is calculated using PCREL_HI20's symbol.
//
// This function returns the R_RISCV_PCREL_HI20 relocation from the
// R_RISCV_PCREL_LO12 relocation.
static Relocation *getRISCVPCRelHi20(Ctx &ctx, const InputSectionBase *loSec,
                                     const Relocation &loReloc) {
  uint64_t addend = loReloc.addend;
  Symbol *sym = loReloc.sym;

  const Defined *d = cast<Defined>(sym);
  if (!d->section) {
    Err(ctx) << loSec->getLocation(loReloc.offset)
             << ": R_RISCV_PCREL_LO12 relocation points to an absolute symbol: "
             << sym->getName();
    return nullptr;
  }
  InputSection *hiSec = cast<InputSection>(d->section);

  if (hiSec != loSec)
    Err(ctx) << loSec->getLocation(loReloc.offset)
             << ": R_RISCV_PCREL_LO12 relocation points to a symbol '"
             << sym->getName() << "' in a different section '" << hiSec->name
             << "'";

  if (addend != 0)
    Warn(ctx) << loSec->getLocation(loReloc.offset)
              << ": non-zero addend in R_RISCV_PCREL_LO12 relocation to "
              << hiSec->getObjMsg(d->value) << " is ignored";

  // Relocations are sorted by offset, so we can use std::equal_range to do
  // binary search.
  Relocation hiReloc;
  hiReloc.offset = d->value;
  auto range =
      std::equal_range(hiSec->relocs().begin(), hiSec->relocs().end(), hiReloc,
                       [](const Relocation &lhs, const Relocation &rhs) {
                         return lhs.offset < rhs.offset;
                       });

  for (auto it = range.first; it != range.second; ++it)
    if (it->type == R_RISCV_PCREL_HI20 || it->type == R_RISCV_GOT_HI20 ||
        it->type == R_RISCV_TLS_GD_HI20 || it->type == R_RISCV_TLS_GOT_HI20)
      return &*it;

  Err(ctx) << loSec->getLocation(loReloc.offset)
           << ": R_RISCV_PCREL_LO12 relocation points to "
           << hiSec->getObjMsg(d->value)
           << " without an associated R_RISCV_PCREL_HI20 relocation";
  return nullptr;
}

// A TLS symbol's virtual address is relative to the TLS segment. Add a
// target-specific adjustment to produce a thread-pointer-relative offset.
static int64_t getTlsTpOffset(Ctx &ctx, const Symbol &s) {
  // On targets that support TLSDESC, _TLS_MODULE_BASE_@tpoff = 0.
  if (&s == ctx.sym.tlsModuleBase)
    return 0;

  // There are 2 TLS layouts. Among targets we support, x86 uses TLS Variant 2
  // while most others use Variant 1. At run time TP will be aligned to p_align.

  // Variant 1. TP will be followed by an optional gap (which is the size of 2
  // pointers on ARM/AArch64, 0 on other targets), followed by alignment
  // padding, then the static TLS blocks. The alignment padding is added so that
  // (TP + gap + padding) is congruent to p_vaddr modulo p_align.
  //
  // Variant 2. Static TLS blocks, followed by alignment padding are placed
  // before TP. The alignment padding is added so that (TP - padding -
  // p_memsz) is congruent to p_vaddr modulo p_align.
  PhdrEntry *tls = ctx.tlsPhdr;
  if (!tls) // Reported an error in getSymVA
    return 0;
  switch (ctx.arg.emachine) {
    // Variant 1.
  case EM_ARM:
  case EM_AARCH64:
    return s.getVA(ctx, 0) + ctx.arg.wordsize * 2 +
           ((tls->p_vaddr - ctx.arg.wordsize * 2) & (tls->p_align - 1));
  case EM_MIPS:
  case EM_PPC:
  case EM_PPC64:
    // Adjusted Variant 1. TP is placed with a displacement of 0x7000, which is
    // to allow a signed 16-bit offset to reach 0x1000 of TCB/thread-library
    // data and 0xf000 of the program's TLS segment.
    return s.getVA(ctx, 0) + (tls->p_vaddr & (tls->p_align - 1)) - 0x7000;
  case EM_LOONGARCH:
  case EM_RISCV:
    // See the comment in handleTlsRelocation. For TLSDESC=>IE,
    // R_RISCV_TLSDESC_{LOAD_LO12,ADD_LO12_I,CALL} also reach here. While
    // `tls` may be null, the return value is ignored.
    if (s.type != STT_TLS)
      return 0;
    return s.getVA(ctx, 0) + (tls->p_vaddr & (tls->p_align - 1));

    // Variant 2.
  case EM_HEXAGON:
  case EM_S390:
  case EM_SPARCV9:
  case EM_386:
  case EM_X86_64:
    return s.getVA(ctx, 0) - tls->p_memsz -
           ((-tls->p_vaddr - tls->p_memsz) & (tls->p_align - 1));
  default:
    llvm_unreachable("unhandled ctx.arg.emachine");
  }
}

uint64_t InputSectionBase::getRelocTargetVA(Ctx &ctx, const Relocation &r,
                                            uint64_t p) const {
  int64_t a = r.addend;
  switch (r.expr) {
  case R_ABS:
  case R_DTPREL:
  case R_RELAX_TLS_LD_TO_LE_ABS:
  case R_RELAX_GOT_PC_NOPIC:
  case RE_AARCH64_AUTH:
  case RE_RISCV_ADD:
  case RE_RISCV_LEB128:
    return r.sym->getVA(ctx, a);
  case R_ADDEND:
    return a;
  case R_RELAX_HINT:
    return 0;
  case RE_ARM_SBREL:
    return r.sym->getVA(ctx, a) - getARMStaticBase(*r.sym);
  case R_GOT:
  case RE_AARCH64_AUTH_GOT:
  case R_RELAX_TLS_GD_TO_IE_ABS:
    return r.sym->getGotVA(ctx) + a;
  case RE_LOONGARCH_GOT:
    // The LoongArch TLS GD relocs reuse the R_LARCH_GOT_PC_LO12 reloc r.type
    // for their page offsets. The arithmetics are different in the TLS case
    // so we have to duplicate some logic here.
    if (r.sym->hasFlag(NEEDS_TLSGD) && r.type != R_LARCH_TLS_IE_PC_LO12)
      // Like RE_LOONGARCH_TLSGD_PAGE_PC but taking the absolute value.
      return ctx.in.got->getGlobalDynAddr(*r.sym) + a;
    return r.sym->getGotVA(ctx) + a;
  case R_GOTONLY_PC:
    return ctx.in.got->getVA() + a - p;
  case R_GOTPLTONLY_PC:
    return ctx.in.gotPlt->getVA() + a - p;
  case R_GOTREL:
  case RE_PPC64_RELAX_TOC:
    return r.sym->getVA(ctx, a) - ctx.in.got->getVA();
  case R_GOTPLTREL:
    return r.sym->getVA(ctx, a) - ctx.in.gotPlt->getVA();
  case R_GOTPLT:
  case R_RELAX_TLS_GD_TO_IE_GOTPLT:
    return r.sym->getGotVA(ctx) + a - ctx.in.gotPlt->getVA();
  case R_TLSLD_GOT_OFF:
  case R_GOT_OFF:
  case R_RELAX_TLS_GD_TO_IE_GOT_OFF:
    return r.sym->getGotOffset(ctx) + a;
  case RE_AARCH64_GOT_PAGE_PC:
  case RE_AARCH64_AUTH_GOT_PAGE_PC:
  case RE_AARCH64_RELAX_TLS_GD_TO_IE_PAGE_PC:
    return getAArch64Page(r.sym->getGotVA(ctx) + a) - getAArch64Page(p);
  case RE_AARCH64_GOT_PAGE:
    return r.sym->getGotVA(ctx) + a - getAArch64Page(ctx.in.got->getVA());
  case R_GOT_PC:
  case RE_AARCH64_AUTH_GOT_PC:
  case R_RELAX_TLS_GD_TO_IE:
    return r.sym->getGotVA(ctx) + a - p;
  case R_GOTPLT_GOTREL:
    return r.sym->getGotPltVA(ctx) + a - ctx.in.got->getVA();
  case R_GOTPLT_PC:
    return r.sym->getGotPltVA(ctx) + a - p;
  case RE_LOONGARCH_GOT_PAGE_PC:
  case RE_LOONGARCH_RELAX_TLS_GD_TO_IE_PAGE_PC:
    if (r.sym->hasFlag(NEEDS_TLSGD))
      return getLoongArchPageDelta(ctx.in.got->getGlobalDynAddr(*r.sym) + a, p,
                                   r.type);
    return getLoongArchPageDelta(r.sym->getGotVA(ctx) + a, p, r.type);
  case RE_MIPS_GOTREL:
    return r.sym->getVA(ctx, a) - ctx.in.mipsGot->getGp(file);
  case RE_MIPS_GOT_GP:
    return ctx.in.mipsGot->getGp(file) + a;
  case RE_MIPS_GOT_GP_PC: {
    // R_MIPS_LO16 expression has RE_MIPS_GOT_GP_PC r.type iif the target
    // is _gp_disp symbol. In that case we should use the following
    // formula for calculation "AHL + GP - P + 4". For details see p. 4-19 at
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    // microMIPS variants of these relocations use slightly different
    // expressions: AHL + GP - P + 3 for %lo() and AHL + GP - P - 1 for %hi()
    // to correctly handle less-significant bit of the microMIPS symbol.
    uint64_t v = ctx.in.mipsGot->getGp(file) + a - p;
    if (r.type == R_MIPS_LO16 || r.type == R_MICROMIPS_LO16)
      v += 4;
    if (r.type == R_MICROMIPS_LO16 || r.type == R_MICROMIPS_HI16)
      v -= 1;
    return v;
  }
  case RE_MIPS_GOT_LOCAL_PAGE:
    // If relocation against MIPS local symbol requires GOT entry, this entry
    // should be initialized by 'page address'. This address is high 16-bits
    // of sum the symbol's value and the addend.
    return ctx.in.mipsGot->getVA() +
           ctx.in.mipsGot->getPageEntryOffset(file, *r.sym, a) -
           ctx.in.mipsGot->getGp(file);
  case RE_MIPS_OSEC_LOCAL_PAGE:
    // This is used by the MIPS multi-GOT implementation. It relocates
    // addresses of 64kb pages that lie inside the output section that sym is
    // a representative for.
    return getMipsPageAddr(r.sym->getOutputSection()->addr) + a;
  case RE_MIPS_GOT_OFF:
  case RE_MIPS_GOT_OFF32:
    // In case of MIPS if a GOT relocation has non-zero addend this addend
    // should be applied to the GOT entry content not to the GOT entry offset.
    // That is why we use separate expression r.type.
    return ctx.in.mipsGot->getVA() +
           ctx.in.mipsGot->getSymEntryOffset(file, *r.sym, a) -
           ctx.in.mipsGot->getGp(file);
  case RE_MIPS_TLSGD:
    return ctx.in.mipsGot->getVA() +
           ctx.in.mipsGot->getGlobalDynOffset(file, *r.sym) -
           ctx.in.mipsGot->getGp(file);
  case RE_MIPS_TLSLD:
    return ctx.in.mipsGot->getVA() + ctx.in.mipsGot->getTlsIndexOffset(file) -
           ctx.in.mipsGot->getGp(file);
  case RE_AARCH64_PAGE_PC: {
    uint64_t val = r.sym->isUndefWeak() ? p + a : r.sym->getVA(ctx, a);
    return getAArch64Page(val) - getAArch64Page(p);
  }
  case RE_RISCV_PC_INDIRECT: {
    if (const Relocation *hiRel = getRISCVPCRelHi20(ctx, this, r))
      return getRelocTargetVA(ctx, *hiRel, r.sym->getVA(ctx));
    return 0;
  }
  case RE_LOONGARCH_PAGE_PC:
    return getLoongArchPageDelta(r.sym->getVA(ctx, a), p, r.type);
  case R_PC:
  case RE_ARM_PCA: {
    uint64_t dest;
    if (r.expr == RE_ARM_PCA)
      // Some PC relative ARM (Thumb) relocations align down the place.
      p = p & 0xfffffffc;
    if (r.sym->isUndefined()) {
      // On ARM and AArch64 a branch to an undefined weak resolves to the next
      // instruction, otherwise the place. On RISC-V, resolve an undefined weak
      // to the same instruction to cause an infinite loop (making the user
      // aware of the issue) while ensuring no overflow.
      // Note: if the symbol is hidden, its binding has been converted to local,
      // so we just check isUndefined() here.
      if (ctx.arg.emachine == EM_ARM)
        dest = getARMUndefinedRelativeWeakVA(r.type, a, p);
      else if (ctx.arg.emachine == EM_AARCH64)
        dest = getAArch64UndefinedRelativeWeakVA(r.type, p) + a;
      else if (ctx.arg.emachine == EM_PPC)
        dest = p;
      else if (ctx.arg.emachine == EM_RISCV)
        dest = getRISCVUndefinedRelativeWeakVA(r.type, p) + a;
      else
        dest = r.sym->getVA(ctx, a);
    } else {
      dest = r.sym->getVA(ctx, a);
    }
    return dest - p;
  }
  case R_PLT:
    return r.sym->getPltVA(ctx) + a;
  case R_PLT_PC:
  case RE_PPC64_CALL_PLT:
    return r.sym->getPltVA(ctx) + a - p;
  case RE_LOONGARCH_PLT_PAGE_PC:
    return getLoongArchPageDelta(r.sym->getPltVA(ctx) + a, p, r.type);
  case R_PLT_GOTPLT:
    return r.sym->getPltVA(ctx) + a - ctx.in.gotPlt->getVA();
  case R_PLT_GOTREL:
    return r.sym->getPltVA(ctx) + a - ctx.in.got->getVA();
  case RE_PPC32_PLTREL:
    // R_PPC_PLTREL24 uses the addend (usually 0 or 0x8000) to indicate r30
    // stores _GLOBAL_OFFSET_TABLE_ or .got2+0x8000. The addend is ignored for
    // target VA computation.
    return r.sym->getPltVA(ctx) - p;
  case RE_PPC64_CALL: {
    uint64_t symVA = r.sym->getVA(ctx, a);
    // If we have an undefined weak symbol, we might get here with a symbol
    // address of zero. That could overflow, but the code must be unreachable,
    // so don't bother doing anything at all.
    if (!symVA)
      return 0;

    // PPC64 V2 ABI describes two entry points to a function. The global entry
    // point is used for calls where the caller and callee (may) have different
    // TOC base pointers and r2 needs to be modified to hold the TOC base for
    // the callee. For local calls the caller and callee share the same
    // TOC base and so the TOC pointer initialization code should be skipped by
    // branching to the local entry point.
    return symVA - p +
           getPPC64GlobalEntryToLocalEntryOffset(ctx, r.sym->stOther);
  }
  case RE_PPC64_TOCBASE:
    return getPPC64TocBase(ctx) + a;
  case R_RELAX_GOT_PC:
  case RE_PPC64_RELAX_GOT_PC:
    return r.sym->getVA(ctx, a) - p;
  case R_RELAX_TLS_GD_TO_LE:
  case R_RELAX_TLS_IE_TO_LE:
  case R_RELAX_TLS_LD_TO_LE:
  case R_TPREL:
    // It is not very clear what to return if the symbol is undefined. With
    // --noinhibit-exec, even a non-weak undefined reference may reach here.
    // Just return A, which matches R_ABS, and the behavior of some dynamic
    // loaders.
    if (r.sym->isUndefined())
      return a;
    return getTlsTpOffset(ctx, *r.sym) + a;
  case R_RELAX_TLS_GD_TO_LE_NEG:
  case R_TPREL_NEG:
    if (r.sym->isUndefined())
      return a;
    return -getTlsTpOffset(ctx, *r.sym) + a;
  case R_SIZE:
    return r.sym->getSize() + a;
  case R_TLSDESC:
  case RE_AARCH64_AUTH_TLSDESC:
    return ctx.in.got->getTlsDescAddr(*r.sym) + a;
  case R_TLSDESC_PC:
    return ctx.in.got->getTlsDescAddr(*r.sym) + a - p;
  case R_TLSDESC_GOTPLT:
    return ctx.in.got->getTlsDescAddr(*r.sym) + a - ctx.in.gotPlt->getVA();
  case RE_AARCH64_TLSDESC_PAGE:
  case RE_AARCH64_AUTH_TLSDESC_PAGE:
    return getAArch64Page(ctx.in.got->getTlsDescAddr(*r.sym) + a) -
           getAArch64Page(p);
  case RE_LOONGARCH_TLSDESC_PAGE_PC:
    return getLoongArchPageDelta(ctx.in.got->getTlsDescAddr(*r.sym) + a, p,
                                 r.type);
  case R_TLSGD_GOT:
    return ctx.in.got->getGlobalDynOffset(*r.sym) + a;
  case R_TLSGD_GOTPLT:
    return ctx.in.got->getGlobalDynAddr(*r.sym) + a - ctx.in.gotPlt->getVA();
  case R_TLSGD_PC:
    return ctx.in.got->getGlobalDynAddr(*r.sym) + a - p;
  case RE_LOONGARCH_TLSGD_PAGE_PC:
    return getLoongArchPageDelta(ctx.in.got->getGlobalDynAddr(*r.sym) + a, p,
                                 r.type);
  case R_TLSLD_GOTPLT:
    return ctx.in.got->getVA() + ctx.in.got->getTlsIndexOff() + a -
           ctx.in.gotPlt->getVA();
  case R_TLSLD_GOT:
    return ctx.in.got->getTlsIndexOff() + a;
  case R_TLSLD_PC:
    return ctx.in.got->getTlsIndexVA() + a - p;
  default:
    llvm_unreachable("invalid expression");
  }
}

// This function applies relocations to sections without SHF_ALLOC bit.
// Such sections are never mapped to memory at runtime. Debug sections are
// an example. Relocations in non-alloc sections are much easier to
// handle than in allocated sections because it will never need complex
// treatment such as GOT or PLT (because at runtime no one refers them).
// So, we handle relocations for non-alloc sections directly in this
// function as a performance optimization.
template <class ELFT, class RelTy>
void InputSection::relocateNonAlloc(Ctx &ctx, uint8_t *buf,
                                    Relocs<RelTy> rels) {
  const unsigned bits = sizeof(typename ELFT::uint) * 8;
  const TargetInfo &target = *ctx.target;
  const auto emachine = ctx.arg.emachine;
  const bool isDebug = isDebugSection(*this);
  const bool isDebugLine = isDebug && name == ".debug_line";
  std::optional<uint64_t> tombstone;
  if (isDebug) {
    if (name == ".debug_loc" || name == ".debug_ranges")
      tombstone = 1;
    else if (name == ".debug_names")
      tombstone = UINT64_MAX; // tombstone value
    else
      tombstone = 0;
  }
  for (const auto &patAndValue : llvm::reverse(ctx.arg.deadRelocInNonAlloc))
    if (patAndValue.first.match(this->name)) {
      tombstone = patAndValue.second;
      break;
    }

  const InputFile *f = this->file;
  for (auto it = rels.begin(), end = rels.end(); it != end; ++it) {
    const RelTy &rel = *it;
    const RelType type = rel.getType(ctx.arg.isMips64EL);
    const uint64_t offset = rel.r_offset;
    uint8_t *bufLoc = buf + offset;
    int64_t addend = getAddend<ELFT>(rel);
    if (!RelTy::HasAddend)
      addend += target.getImplicitAddend(bufLoc, type);

    Symbol &sym = f->getRelocTargetSym(rel);
    RelExpr expr = target.getRelExpr(type, sym, bufLoc);
    if (expr == R_NONE)
      continue;
    auto *ds = dyn_cast<Defined>(&sym);

    if (emachine == EM_RISCV && type == R_RISCV_SET_ULEB128) {
      if (++it != end &&
          it->getType(/*isMips64EL=*/false) == R_RISCV_SUB_ULEB128 &&
          it->r_offset == offset) {
        uint64_t val;
        if (!ds && tombstone) {
          val = *tombstone;
        } else {
          val = sym.getVA(ctx, addend) -
                (f->getRelocTargetSym(*it).getVA(ctx) + getAddend<ELFT>(*it));
        }
        if (overwriteULEB128(bufLoc, val) >= 0x80)
          Err(ctx) << getLocation(offset) << ": ULEB128 value " << val
                   << " exceeds available space; references '" << &sym << "'";
        continue;
      }
      Err(ctx) << getLocation(offset)
               << ": R_RISCV_SET_ULEB128 not paired with R_RISCV_SUB_SET128";
      return;
    }

    if (tombstone && (expr == R_ABS || expr == R_DTPREL)) {
      // Resolve relocations in .debug_* referencing (discarded symbols or ICF
      // folded section symbols) to a tombstone value. Resolving to addend is
      // unsatisfactory because the result address range may collide with a
      // valid range of low address, or leave multiple CUs claiming ownership of
      // the same range of code, which may confuse consumers.
      //
      // To address the problems, we use -1 as a tombstone value for most
      // .debug_* sections. We have to ignore the addend because we don't want
      // to resolve an address attribute (which may have a non-zero addend) to
      // -1+addend (wrap around to a low address).
      //
      // R_DTPREL type relocations represent an offset into the dynamic thread
      // vector. The computed value is st_value plus a non-negative offset.
      // Negative values are invalid, so -1 can be used as the tombstone value.
      //
      // If the referenced symbol is relative to a discarded section (due to
      // --gc-sections, COMDAT, etc), it has been converted to a Undefined.
      // `ds->folded` catches the ICF folded case. However, resolving a
      // relocation in .debug_line to -1 would stop debugger users from setting
      // breakpoints on the folded-in function, so exclude .debug_line.
      //
      // For pre-DWARF-v5 .debug_loc and .debug_ranges, -1 is a reserved value
      // (base address selection entry), use 1 (which is used by GNU ld for
      // .debug_ranges).
      //
      // TODO To reduce disruption, we use 0 instead of -1 as the tombstone
      // value. Enable -1 in a future release.
      if (!ds || (ds->folded && !isDebugLine)) {
        // If -z dead-reloc-in-nonalloc= is specified, respect it.
        uint64_t value = SignExtend64<bits>(*tombstone);
        // For a 32-bit local TU reference in .debug_names, X86_64::relocate
        // requires that the unsigned value for R_X86_64_32 is truncated to
        // 32-bit. Other 64-bit targets's don't discern signed/unsigned 32-bit
        // absolute relocations and do not need this change.
        if (emachine == EM_X86_64 && type == R_X86_64_32)
          value = static_cast<uint32_t>(value);
        target.relocateNoSym(bufLoc, type, value);
        continue;
      }
    }

    // For a relocatable link, content relocated by relocation types with an
    // explicit addend, such as RELA, remain unchanged and we can stop here.
    // While content relocated by relocation types with an implicit addend, such
    // as REL, needs the implicit addend updated.
    if (ctx.arg.relocatable && (RelTy::HasAddend || sym.type != STT_SECTION))
      continue;

    // R_ABS/R_DTPREL and some other relocations can be used from non-SHF_ALLOC
    // sections.
    if (LLVM_LIKELY(expr == R_ABS) || expr == R_DTPREL || expr == R_GOTPLTREL ||
        expr == RE_RISCV_ADD || expr == RE_ARM_SBREL) {
      target.relocateNoSym(bufLoc, type,
                           SignExtend64<bits>(sym.getVA(ctx, addend)));
      continue;
    }

    if (expr == R_SIZE) {
      target.relocateNoSym(bufLoc, type,
                           SignExtend64<bits>(sym.getSize() + addend));
      continue;
    }

    // If the control reaches here, we found a PC-relative relocation in a
    // non-ALLOC section. Since non-ALLOC section is not loaded into memory
    // at runtime, the notion of PC-relative doesn't make sense here. So,
    // this is a usage error. However, GNU linkers historically accept such
    // relocations without any errors and relocate them as if they were at
    // address 0. For bug-compatibility, we accept them with warnings. We
    // know Steel Bank Common Lisp as of 2018 have this bug.
    //
    // GCC 8.0 or earlier have a bug that they emit R_386_GOTPC relocations
    // against _GLOBAL_OFFSET_TABLE_ for .debug_info. The bug has been fixed in
    // 2017 (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=82630), but we need to
    // keep this bug-compatible code for a while.
    bool isErr = expr != R_PC && !(emachine == EM_386 && type == R_386_GOTPC);
    {
      ELFSyncStream diag(ctx, isErr && !ctx.arg.noinhibitExec
                                  ? DiagLevel::Err
                                  : DiagLevel::Warn);
      diag << getLocation(offset) << ": has non-ABS relocation " << type
           << " against symbol '" << &sym << "'";
    }
    if (!isErr)
      target.relocateNoSym(
          bufLoc, type,
          SignExtend64<bits>(sym.getVA(ctx, addend - offset - outSecOff)));
  }
}

template <class ELFT>
void InputSectionBase::relocate(Ctx &ctx, uint8_t *buf, uint8_t *bufEnd) {
  if ((flags & SHF_EXECINSTR) && LLVM_UNLIKELY(getFile<ELFT>()->splitStack))
    adjustSplitStackFunctionPrologues<ELFT>(ctx, buf, bufEnd);

  if (flags & SHF_ALLOC) {
    ctx.target->relocateAlloc(*this, buf);
    return;
  }

  auto *sec = cast<InputSection>(this);
  // For a relocatable link, also call relocateNonAlloc() to rewrite applicable
  // locations with tombstone values.
  invokeOnRelocs(*sec, sec->relocateNonAlloc<ELFT>, ctx, buf);
}

// For each function-defining prologue, find any calls to __morestack,
// and replace them with calls to __morestack_non_split.
static void switchMorestackCallsToMorestackNonSplit(
    Ctx &ctx, DenseSet<Defined *> &prologues,
    SmallVector<Relocation *, 0> &morestackCalls) {

  // If the target adjusted a function's prologue, all calls to
  // __morestack inside that function should be switched to
  // __morestack_non_split.
  Symbol *moreStackNonSplit = ctx.symtab->find("__morestack_non_split");
  if (!moreStackNonSplit) {
    ErrAlways(ctx) << "mixing split-stack objects requires a definition of "
                      "__morestack_non_split";
    return;
  }

  // Sort both collections to compare addresses efficiently.
  llvm::sort(morestackCalls, [](const Relocation *l, const Relocation *r) {
    return l->offset < r->offset;
  });
  std::vector<Defined *> functions(prologues.begin(), prologues.end());
  llvm::sort(functions, [](const Defined *l, const Defined *r) {
    return l->value < r->value;
  });

  auto it = morestackCalls.begin();
  for (Defined *f : functions) {
    // Find the first call to __morestack within the function.
    while (it != morestackCalls.end() && (*it)->offset < f->value)
      ++it;
    // Adjust all calls inside the function.
    while (it != morestackCalls.end() && (*it)->offset < f->value + f->size) {
      (*it)->sym = moreStackNonSplit;
      ++it;
    }
  }
}

static bool enclosingPrologueAttempted(uint64_t offset,
                                       const DenseSet<Defined *> &prologues) {
  for (Defined *f : prologues)
    if (f->value <= offset && offset < f->value + f->size)
      return true;
  return false;
}

// If a function compiled for split stack calls a function not
// compiled for split stack, then the caller needs its prologue
// adjusted to ensure that the called function will have enough stack
// available. Find those functions, and adjust their prologues.
template <class ELFT>
void InputSectionBase::adjustSplitStackFunctionPrologues(Ctx &ctx, uint8_t *buf,
                                                         uint8_t *end) {
  DenseSet<Defined *> prologues;
  SmallVector<Relocation *, 0> morestackCalls;

  for (Relocation &rel : relocs()) {
    // Ignore calls into the split-stack api.
    if (rel.sym->getName().starts_with("__morestack")) {
      if (rel.sym->getName() == "__morestack")
        morestackCalls.push_back(&rel);
      continue;
    }

    // A relocation to non-function isn't relevant. Sometimes
    // __morestack is not marked as a function, so this check comes
    // after the name check.
    if (rel.sym->type != STT_FUNC)
      continue;

    // If the callee's-file was compiled with split stack, nothing to do.  In
    // this context, a "Defined" symbol is one "defined by the binary currently
    // being produced". So an "undefined" symbol might be provided by a shared
    // library. It is not possible to tell how such symbols were compiled, so be
    // conservative.
    if (Defined *d = dyn_cast<Defined>(rel.sym))
      if (InputSection *isec = cast_or_null<InputSection>(d->section))
        if (!isec || !isec->getFile<ELFT>() || isec->getFile<ELFT>()->splitStack)
          continue;

    if (enclosingPrologueAttempted(rel.offset, prologues))
      continue;

    if (Defined *f = getEnclosingFunction(rel.offset)) {
      prologues.insert(f);
      if (ctx.target->adjustPrologueForCrossSplitStack(buf + f->value, end,
                                                       f->stOther))
        continue;
      if (!getFile<ELFT>()->someNoSplitStack)
        Err(ctx)
            << this << ": " << f->getName() << " (with -fsplit-stack) calls "
            << rel.sym->getName()
            << " (without -fsplit-stack), but couldn't adjust its prologue";
    }
  }

  if (ctx.target->needsMoreStackNonSplit)
    switchMorestackCallsToMorestackNonSplit(ctx, prologues, morestackCalls);
}

template <class ELFT> void InputSection::writeTo(Ctx &ctx, uint8_t *buf) {
  if (LLVM_UNLIKELY(type == SHT_NOBITS))
    return;
  // If -r or --emit-relocs is given, then an InputSection
  // may be a relocation section.
  if (LLVM_UNLIKELY(type == SHT_RELA)) {
    copyRelocations<ELFT, typename ELFT::Rela>(ctx, buf);
    return;
  }
  if (LLVM_UNLIKELY(type == SHT_REL)) {
    copyRelocations<ELFT, typename ELFT::Rel>(ctx, buf);
    return;
  }

  // If -r is given, we may have a SHT_GROUP section.
  if (LLVM_UNLIKELY(type == SHT_GROUP)) {
    copyShtGroup<ELFT>(buf);
    return;
  }

  // If this is a compressed section, uncompress section contents directly
  // to the buffer.
  if (compressed) {
    auto *hdr = reinterpret_cast<const typename ELFT::Chdr *>(content_);
    auto compressed = ArrayRef<uint8_t>(content_, compressedSize)
                          .slice(sizeof(typename ELFT::Chdr));
    size_t size = this->size;
    if (Error e = hdr->ch_type == ELFCOMPRESS_ZLIB
                      ? compression::zlib::decompress(compressed, buf, size)
                      : compression::zstd::decompress(compressed, buf, size))
      Err(ctx) << this << ": decompress failed: " << std::move(e);
    uint8_t *bufEnd = buf + size;
    relocate<ELFT>(ctx, buf, bufEnd);
    return;
  }

  // Copy section contents from source object file to output file
  // and then apply relocations.
  memcpy(buf, content().data(), content().size());
  relocate<ELFT>(ctx, buf, buf + content().size());
}

void InputSection::replace(InputSection *other) {
  addralign = std::max(addralign, other->addralign);

  // When a section is replaced with another section that was allocated to
  // another partition, the replacement section (and its associated sections)
  // need to be placed in the main partition so that both partitions will be
  // able to access it.
  if (partition != other->partition) {
    partition = 1;
    for (InputSection *isec : dependentSections)
      isec->partition = 1;
  }

  other->repl = repl;
  other->markDead();
}

template <class ELFT>
EhInputSection::EhInputSection(ObjFile<ELFT> &f,
                               const typename ELFT::Shdr &header,
                               StringRef name)
    : InputSectionBase(f, header, name, InputSectionBase::EHFrame) {}

SyntheticSection *EhInputSection::getParent() const {
  return cast_or_null<SyntheticSection>(parent);
}

// .eh_frame is a sequence of CIE or FDE records.
// This function splits an input section into records and returns them.
template <class ELFT> void EhInputSection::split() {
  const RelsOrRelas<ELFT> rels = relsOrRelas<ELFT>(/*supportsCrel=*/false);
  // getReloc expects the relocations to be sorted by r_offset. See the comment
  // in scanRelocs.
  if (rels.areRelocsRel()) {
    SmallVector<typename ELFT::Rel, 0> storage;
    split<ELFT>(sortRels(rels.rels, storage));
  } else {
    SmallVector<typename ELFT::Rela, 0> storage;
    split<ELFT>(sortRels(rels.relas, storage));
  }
}

template <class ELFT, class RelTy>
void EhInputSection::split(ArrayRef<RelTy> rels) {
  ArrayRef<uint8_t> d = content();
  const char *msg = nullptr;
  unsigned relI = 0;
  while (!d.empty()) {
    if (d.size() < 4) {
      msg = "CIE/FDE too small";
      break;
    }
    uint64_t size = endian::read32<ELFT::Endianness>(d.data());
    if (size == 0) // ZERO terminator
      break;
    uint32_t id = endian::read32<ELFT::Endianness>(d.data() + 4);
    size += 4;
    if (LLVM_UNLIKELY(size > d.size())) {
      // If it is 0xFFFFFFFF, the next 8 bytes contain the size instead,
      // but we do not support that format yet.
      msg = size == UINT32_MAX + uint64_t(4)
                ? "CIE/FDE too large"
                : "CIE/FDE ends past the end of the section";
      break;
    }

    // Find the first relocation that points to [off,off+size). Relocations
    // have been sorted by r_offset.
    const uint64_t off = d.data() - content().data();
    while (relI != rels.size() && rels[relI].r_offset < off)
      ++relI;
    unsigned firstRel = -1;
    if (relI != rels.size() && rels[relI].r_offset < off + size)
      firstRel = relI;
    (id == 0 ? cies : fdes).emplace_back(off, this, size, firstRel);
    d = d.slice(size);
  }
  if (msg)
    Err(file->ctx) << "corrupted .eh_frame: " << msg << "\n>>> defined in "
                   << getObjMsg(d.data() - content().data());
}

// Return the offset in an output section for a given input offset.
uint64_t EhInputSection::getParentOffset(uint64_t offset) const {
  auto it = partition_point(
      fdes, [=](EhSectionPiece p) { return p.inputOff <= offset; });
  if (it == fdes.begin() || it[-1].inputOff + it[-1].size <= offset) {
    it = partition_point(
        cies, [=](EhSectionPiece p) { return p.inputOff <= offset; });
    if (it == cies.begin()) // invalid piece
      return offset;
  }
  if (it[-1].outputOff == -1) // invalid piece
    return offset - it[-1].inputOff;
  return it[-1].outputOff + (offset - it[-1].inputOff);
}

static size_t findNull(StringRef s, size_t entSize) {
  for (unsigned i = 0, n = s.size(); i != n; i += entSize) {
    const char *b = s.begin() + i;
    if (std::all_of(b, b + entSize, [](char c) { return c == 0; }))
      return i;
  }
  llvm_unreachable("");
}

// Split SHF_STRINGS section. Such section is a sequence of
// null-terminated strings.
void MergeInputSection::splitStrings(StringRef s, size_t entSize) {
  const bool live = !(flags & SHF_ALLOC) || !getCtx().arg.gcSections;
  const char *p = s.data(), *end = s.data() + s.size();
  if (!std::all_of(end - entSize, end, [](char c) { return c == 0; })) {
    Err(getCtx()) << this << ": string is not null terminated";
    pieces.emplace_back(entSize, 0, false);
    return;
  }
  if (entSize == 1) {
    // Optimize the common case.
    do {
      size_t size = strlen(p);
      pieces.emplace_back(p - s.begin(), xxh3_64bits(StringRef(p, size)), live);
      p += size + 1;
    } while (p != end);
  } else {
    do {
      size_t size = findNull(StringRef(p, end - p), entSize);
      pieces.emplace_back(p - s.begin(), xxh3_64bits(StringRef(p, size)), live);
      p += size + entSize;
    } while (p != end);
  }
}

// Split non-SHF_STRINGS section. Such section is a sequence of
// fixed size records.
void MergeInputSection::splitNonStrings(ArrayRef<uint8_t> data,
                                        size_t entSize) {
  size_t size = data.size();
  assert((size % entSize) == 0);
  const bool live = !(flags & SHF_ALLOC) || !getCtx().arg.gcSections;

  pieces.resize_for_overwrite(size / entSize);
  for (size_t i = 0, j = 0; i != size; i += entSize, j++)
    pieces[j] = {i, (uint32_t)xxh3_64bits(data.slice(i, entSize)), live};
}

template <class ELFT>
MergeInputSection::MergeInputSection(ObjFile<ELFT> &f,
                                     const typename ELFT::Shdr &header,
                                     StringRef name)
    : InputSectionBase(f, header, name, InputSectionBase::Merge) {}

MergeInputSection::MergeInputSection(Ctx &ctx, StringRef name, uint32_t type,
                                     uint64_t flags, uint64_t entsize,
                                     ArrayRef<uint8_t> data)
    : InputSectionBase(ctx.internalFile, name, type, flags, /*link=*/0,
                       /*info=*/0,
                       /*addralign=*/entsize, entsize, data,
                       SectionBase::Merge) {}

// This function is called after we obtain a complete list of input sections
// that need to be linked. This is responsible to split section contents
// into small chunks for further processing.
//
// Note that this function is called from parallelForEach. This must be
// thread-safe (i.e. no memory allocation from the pools).
void MergeInputSection::splitIntoPieces() {
  assert(pieces.empty());

  if (flags & SHF_STRINGS)
    splitStrings(toStringRef(contentMaybeDecompress()), entsize);
  else
    splitNonStrings(contentMaybeDecompress(), entsize);
}

SectionPiece &MergeInputSection::getSectionPiece(uint64_t offset) {
  if (content().size() <= offset) {
    Err(getCtx()) << this << ": offset is outside the section";
    return pieces[0];
  }
  return partition_point(
      pieces, [=](SectionPiece p) { return p.inputOff <= offset; })[-1];
}

// Return the offset in an output section for a given input offset.
uint64_t MergeInputSection::getParentOffset(uint64_t offset) const {
  const SectionPiece &piece = getSectionPiece(offset);
  return piece.outputOff + (offset - piece.inputOff);
}

template InputSection::InputSection(ObjFile<ELF32LE> &, const ELF32LE::Shdr &,
                                    StringRef);
template InputSection::InputSection(ObjFile<ELF32BE> &, const ELF32BE::Shdr &,
                                    StringRef);
template InputSection::InputSection(ObjFile<ELF64LE> &, const ELF64LE::Shdr &,
                                    StringRef);
template InputSection::InputSection(ObjFile<ELF64BE> &, const ELF64BE::Shdr &,
                                    StringRef);

template void InputSection::writeTo<ELF32LE>(Ctx &, uint8_t *);
template void InputSection::writeTo<ELF32BE>(Ctx &, uint8_t *);
template void InputSection::writeTo<ELF64LE>(Ctx &, uint8_t *);
template void InputSection::writeTo<ELF64BE>(Ctx &, uint8_t *);

template RelsOrRelas<ELF32LE>
InputSectionBase::relsOrRelas<ELF32LE>(bool) const;
template RelsOrRelas<ELF32BE>
InputSectionBase::relsOrRelas<ELF32BE>(bool) const;
template RelsOrRelas<ELF64LE>
InputSectionBase::relsOrRelas<ELF64LE>(bool) const;
template RelsOrRelas<ELF64BE>
InputSectionBase::relsOrRelas<ELF64BE>(bool) const;

template MergeInputSection::MergeInputSection(ObjFile<ELF32LE> &,
                                              const ELF32LE::Shdr &, StringRef);
template MergeInputSection::MergeInputSection(ObjFile<ELF32BE> &,
                                              const ELF32BE::Shdr &, StringRef);
template MergeInputSection::MergeInputSection(ObjFile<ELF64LE> &,
                                              const ELF64LE::Shdr &, StringRef);
template MergeInputSection::MergeInputSection(ObjFile<ELF64BE> &,
                                              const ELF64BE::Shdr &, StringRef);

template EhInputSection::EhInputSection(ObjFile<ELF32LE> &,
                                        const ELF32LE::Shdr &, StringRef);
template EhInputSection::EhInputSection(ObjFile<ELF32BE> &,
                                        const ELF32BE::Shdr &, StringRef);
template EhInputSection::EhInputSection(ObjFile<ELF64LE> &,
                                        const ELF64LE::Shdr &, StringRef);
template EhInputSection::EhInputSection(ObjFile<ELF64BE> &,
                                        const ELF64BE::Shdr &, StringRef);

template void EhInputSection::split<ELF32LE>();
template void EhInputSection::split<ELF32BE>();
template void EhInputSection::split<ELF64LE>();
template void EhInputSection::split<ELF64BE>();
