//===- InputSection.cpp ---------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "Config.h"
#include "EhFrame.h"
#include "Error.h"
#include "InputFiles.h"
#include "LinkerScript.h"
#include "Memory.h"
#include "OutputSections.h"
#include "Relocations.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Thunks.h"
#include "llvm/Object/Decompressor.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Path.h"
#include <mutex>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support;
using namespace llvm::support::endian;
using namespace llvm::sys;

using namespace lld;
using namespace lld::elf;

std::vector<InputSectionBase *> elf::InputSections;

// Returns a string to construct an error message.
std::string lld::toString(const InputSectionBase *Sec) {
  // File can be absent if section is synthetic.
  std::string FileName = Sec->File ? Sec->File->getName() : "<internal>";
  return (FileName + ":(" + Sec->Name + ")").str();
}

template <class ELFT>
static ArrayRef<uint8_t> getSectionContents(elf::ObjectFile<ELFT> *File,
                                            const typename ELFT::Shdr *Hdr) {
  if (!File || Hdr->sh_type == SHT_NOBITS)
    return makeArrayRef<uint8_t>(nullptr, Hdr->sh_size);
  return check(File->getObj().getSectionContents(Hdr));
}

InputSectionBase::InputSectionBase(InputFile *File, uint64_t Flags,
                                   uint32_t Type, uint64_t Entsize,
                                   uint32_t Link, uint32_t Info,
                                   uint32_t Alignment, ArrayRef<uint8_t> Data,
                                   StringRef Name, Kind SectionKind)
    : SectionBase(SectionKind, Name, Flags, Entsize, Alignment, Type, Info,
                  Link),
      File(File), Data(Data), Repl(this) {
  Live = !Config->GcSections || !(Flags & SHF_ALLOC);
  Assigned = false;
  NumRelocations = 0;
  AreRelocsRela = false;

  // The ELF spec states that a value of 0 means the section has
  // no alignment constraits.
  uint32_t V = std::max<uint64_t>(Alignment, 1);
  if (!isPowerOf2_64(V))
    fatal(toString(File) + ": section sh_addralign is not a power of 2");
  this->Alignment = V;
}

// GNU assembler 2.24 and LLVM 4.0.0's MC (the newest release as of
// March 2017) fail to infer section types for sections starting with
// ".init_array." or ".fini_array.". They set SHT_PROGBITS instead of
// SHF_INIT_ARRAY. As a result, the following assembler directive
// creates ".init_array.100" with SHT_PROGBITS, for example.
//
//   .section .init_array.100, "aw"
//
// This function forces SHT_{INIT,FINI}_ARRAY so that we can handle
// incorrect inputs as if they were correct from the beginning.
static uint64_t getType(uint64_t Type, StringRef Name) {
  if (Type == SHT_PROGBITS && Name.startswith(".init_array."))
    return SHT_INIT_ARRAY;
  if (Type == SHT_PROGBITS && Name.startswith(".fini_array."))
    return SHT_FINI_ARRAY;
  return Type;
}

template <class ELFT>
InputSectionBase::InputSectionBase(elf::ObjectFile<ELFT> *File,
                                   const typename ELFT::Shdr *Hdr,
                                   StringRef Name, Kind SectionKind)
    : InputSectionBase(File, Hdr->sh_flags & ~SHF_INFO_LINK,
                       getType(Hdr->sh_type, Name), Hdr->sh_entsize,
                       Hdr->sh_link, Hdr->sh_info, Hdr->sh_addralign,
                       getSectionContents(File, Hdr), Name, SectionKind) {
  // We reject object files having insanely large alignments even though
  // they are allowed by the spec. I think 4GB is a reasonable limitation.
  // We might want to relax this in the future.
  if (Hdr->sh_addralign > UINT32_MAX)
    fatal(toString(File) + ": section sh_addralign is too large");
}

size_t InputSectionBase::getSize() const {
  if (auto *S = dyn_cast<SyntheticSection>(this))
    return S->getSize();

  return Data.size();
}

uint64_t InputSectionBase::getOffsetInFile() const {
  const uint8_t *FileStart = (const uint8_t *)File->MB.getBufferStart();
  const uint8_t *SecStart = Data.begin();
  return SecStart - FileStart;
}

uint64_t SectionBase::getOffset(uint64_t Offset) const {
  switch (kind()) {
  case Output: {
    auto *OS = cast<OutputSection>(this);
    // For output sections we treat offset -1 as the end of the section.
    return Offset == uint64_t(-1) ? OS->Size : Offset;
  }
  case Regular:
    return cast<InputSection>(this)->OutSecOff + Offset;
  case Synthetic: {
    auto *IS = cast<InputSection>(this);
    // For synthetic sections we treat offset -1 as the end of the section.
    return IS->OutSecOff + (Offset == uint64_t(-1) ? IS->getSize() : Offset);
  }
  case EHFrame:
    // The file crtbeginT.o has relocations pointing to the start of an empty
    // .eh_frame that is known to be the first in the link. It does that to
    // identify the start of the output .eh_frame.
    return Offset;
  case Merge:
    const MergeInputSection *MS = cast<MergeInputSection>(this);
    if (MS->MergeSec)
      return MS->MergeSec->OutSecOff + MS->getOffset(Offset);
    return MS->getOffset(Offset);
  }
  llvm_unreachable("invalid section kind");
}

OutputSection *SectionBase::getOutputSection() {
  if (auto *IS = dyn_cast<InputSection>(this))
    return IS->OutSec;
  if (auto *MS = dyn_cast<MergeInputSection>(this))
    return MS->MergeSec ? MS->MergeSec->OutSec : nullptr;
  if (auto *EH = dyn_cast<EhInputSection>(this))
    return EH->EHSec->OutSec;
  return cast<OutputSection>(this);
}

// Uncompress section contents. Note that this function is called
// from parallel_for_each, so it must be thread-safe.
void InputSectionBase::uncompress() {
  Decompressor Dec = check(Decompressor::create(Name, toStringRef(Data),
                                                Config->IsLE, Config->Is64));

  size_t Size = Dec.getDecompressedSize();
  char *OutputBuf;
  {
    static std::mutex Mu;
    std::lock_guard<std::mutex> Lock(Mu);
    OutputBuf = BAlloc.Allocate<char>(Size);
  }

  if (Error E = Dec.decompress({OutputBuf, Size}))
    fatal(toString(this) +
          ": decompress failed: " + llvm::toString(std::move(E)));
  Data = ArrayRef<uint8_t>((uint8_t *)OutputBuf, Size);
}

uint64_t SectionBase::getOffset(const DefinedRegular &Sym) const {
  return getOffset(Sym.Value);
}

InputSectionBase *InputSectionBase::getLinkOrderDep() const {
  if ((Flags & SHF_LINK_ORDER) && Link != 0)
    return File->getSections()[Link];
  return nullptr;
}

// Returns a source location string. Used to construct an error message.
template <class ELFT>
std::string InputSectionBase::getLocation(uint64_t Offset) {
  // We don't have file for synthetic sections.
  if (getFile<ELFT>() == nullptr)
    return (Config->OutputFile + ":(" + Name + "+0x" + utohexstr(Offset) + ")")
        .str();

  // First check if we can get desired values from debugging information.
  std::string LineInfo = getFile<ELFT>()->getLineInfo(this, Offset);
  if (!LineInfo.empty())
    return LineInfo;

  // File->SourceFile contains STT_FILE symbol that contains a
  // source file name. If it's missing, we use an object file name.
  std::string SrcFile = getFile<ELFT>()->SourceFile;
  if (SrcFile.empty())
    SrcFile = toString(File);

  // Find a function symbol that encloses a given location.
  for (SymbolBody *B : getFile<ELFT>()->getSymbols())
    if (auto *D = dyn_cast<DefinedRegular>(B))
      if (D->Section == this && D->Type == STT_FUNC)
        if (D->Value <= Offset && Offset < D->Value + D->Size)
          return SrcFile + ":(function " + toString(*D) + ")";

  // If there's no symbol, print out the offset in the section.
  return (SrcFile + ":(" + Name + "+0x" + utohexstr(Offset) + ")").str();
}

// Returns a source location string. This function is intended to be
// used for constructing an error message. The returned message looks
// like this:
//
//   foo.c:42 (/home/alice/possibly/very/long/path/foo.c:42)
//
// Returns an empty string if there's no way to get line info.
template <class ELFT> std::string InputSectionBase::getSrcMsg(uint64_t Offset) {
  // Synthetic sections don't have input files.
  elf::ObjectFile<ELFT> *File = getFile<ELFT>();
  if (!File)
    return "";

  Optional<DILineInfo> Info = File->getDILineInfo(this, Offset);

  // File->SourceFile contains STT_FILE symbol, and that is a last resort.
  if (!Info)
    return File->SourceFile;

  std::string Path = Info->FileName;
  std::string Filename = path::filename(Path);
  std::string Lineno = ":" + std::to_string(Info->Line);
  if (Filename == Path)
    return Filename + Lineno;
  return Filename + Lineno + " (" + Path + Lineno + ")";
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
template <class ELFT> std::string InputSectionBase::getObjMsg(uint64_t Off) {
  // Synthetic sections don't have input files.
  elf::ObjectFile<ELFT> *File = getFile<ELFT>();
  std::string Filename = File ? File->getName() : "(internal)";

  std::string Archive;
  if (!File->ArchiveName.empty())
    Archive = (" in archive " + File->ArchiveName).str();

  // Find a symbol that encloses a given location.
  for (SymbolBody *B : getFile<ELFT>()->getSymbols())
    if (auto *D = dyn_cast<DefinedRegular>(B))
      if (D->Section == this && D->Value <= Off && Off < D->Value + D->Size)
        return Filename + ":(" + toString(*D) + ")" + Archive;

  // If there's no symbol, print out the offset in the section.
  return (Filename + ":(" + Name + "+0x" + utohexstr(Off) + ")" + Archive)
      .str();
}

InputSectionBase InputSectionBase::Discarded;

InputSection::InputSection(uint64_t Flags, uint32_t Type, uint32_t Alignment,
                           ArrayRef<uint8_t> Data, StringRef Name, Kind K)
    : InputSectionBase(nullptr, Flags, Type,
                       /*Entsize*/ 0, /*Link*/ 0, /*Info*/ 0, Alignment, Data,
                       Name, K) {}

template <class ELFT>
InputSection::InputSection(elf::ObjectFile<ELFT> *F,
                           const typename ELFT::Shdr *Header, StringRef Name)
    : InputSectionBase(F, Header, Name, InputSectionBase::Regular) {}

bool InputSection::classof(const SectionBase *S) {
  return S->kind() == SectionBase::Regular ||
         S->kind() == SectionBase::Synthetic;
}

bool InputSectionBase::classof(const SectionBase *S) {
  return S->kind() != Output;
}

InputSectionBase *InputSection::getRelocatedSection() {
  assert(this->Type == SHT_RELA || this->Type == SHT_REL);
  ArrayRef<InputSectionBase *> Sections = this->File->getSections();
  return Sections[this->Info];
}

// This is used for -r and --emit-relocs. We can't use memcpy to copy
// relocations because we need to update symbol table offset and section index
// for each relocation. So we copy relocations one by one.
template <class ELFT, class RelTy>
void InputSection::copyRelocations(uint8_t *Buf, ArrayRef<RelTy> Rels) {
  InputSectionBase *RelocatedSection = getRelocatedSection();

  // Loop is slow and have complexity O(N*M), where N - amount of
  // relocations and M - amount of symbols in symbol table.
  // That happens because getSymbolIndex(...) call below performs
  // simple linear search.
  for (const RelTy &Rel : Rels) {
    uint32_t Type = Rel.getType(Config->IsMips64EL);
    SymbolBody &Body = this->getFile<ELFT>()->getRelocTargetSym(Rel);

    auto *P = reinterpret_cast<typename ELFT::Rela *>(Buf);
    Buf += sizeof(RelTy);

    if (Config->IsRela)
      P->r_addend = getAddend<ELFT>(Rel);

    // Output section VA is zero for -r, so r_offset is an offset within the
    // section, but for --emit-relocs it is an virtual address.
    P->r_offset = RelocatedSection->OutSec->Addr +
                  RelocatedSection->getOffset(Rel.r_offset);
    P->setSymbolAndType(In<ELFT>::SymTab->getSymbolIndex(&Body), Type,
                        Config->IsMips64EL);

    if (Body.Type == STT_SECTION) {
      // We combine multiple section symbols into only one per
      // section. This means we have to update the addend. That is
      // trivial for Elf_Rela, but for Elf_Rel we have to write to the
      // section data. We do that by adding to the Relocation vector.

      // .eh_frame is horribly special and can reference discarded sections. To
      // avoid having to parse and recreate .eh_frame, we just replace any
      // relocation in it pointing to discarded sections with R_*_NONE, which
      // hopefully creates a frame that is ignored at runtime.
      SectionBase *Section = cast<DefinedRegular>(Body).Section;
      if (Section == &InputSection::Discarded) {
        P->setSymbolAndType(0, 0, false);
        continue;
      }

      if (Config->IsRela) {
        P->r_addend += Body.getVA() - Section->getOutputSection()->Addr;
      } else if (Config->Relocatable) {
        const uint8_t *BufLoc = RelocatedSection->Data.begin() + Rel.r_offset;
        RelocatedSection->Relocations.push_back(
            {R_ABS, Type, Rel.r_offset, Target->getImplicitAddend(BufLoc, Type),
             &Body});
      }
    }

  }
}

static uint32_t getARMUndefinedRelativeWeakVA(uint32_t Type, uint32_t A,
                                              uint32_t P) {
  switch (Type) {
  case R_ARM_THM_JUMP11:
    return P + 2;
  case R_ARM_CALL:
  case R_ARM_JUMP24:
  case R_ARM_PC24:
  case R_ARM_PLT32:
  case R_ARM_PREL31:
  case R_ARM_THM_JUMP19:
  case R_ARM_THM_JUMP24:
    return P + 4;
  case R_ARM_THM_CALL:
    // We don't want an interworking BLX to ARM
    return P + 5;
  default:
    return A;
  }
}

static uint64_t getAArch64UndefinedRelativeWeakVA(uint64_t Type, uint64_t A,
                                                  uint64_t P) {
  switch (Type) {
  case R_AARCH64_CALL26:
  case R_AARCH64_CONDBR19:
  case R_AARCH64_JUMP26:
  case R_AARCH64_TSTBR14:
    return P + 4;
  default:
    return A;
  }
}

template <class ELFT>
static typename ELFT::uint
getRelocTargetVA(uint32_t Type, int64_t A, typename ELFT::uint P,
                 const SymbolBody &Body, RelExpr Expr) {
  switch (Expr) {
  case R_ABS:
  case R_RELAX_GOT_PC_NOPIC:
    return Body.getVA(A);
  case R_GOT:
  case R_RELAX_TLS_GD_TO_IE_ABS:
    return Body.getGotVA<ELFT>() + A;
  case R_GOTONLY_PC:
    return In<ELFT>::Got->getVA() + A - P;
  case R_GOTONLY_PC_FROM_END:
    return In<ELFT>::Got->getVA() + A - P + In<ELFT>::Got->getSize();
  case R_GOTREL:
    return Body.getVA(A) - In<ELFT>::Got->getVA();
  case R_GOTREL_FROM_END:
    return Body.getVA(A) - In<ELFT>::Got->getVA() - In<ELFT>::Got->getSize();
  case R_GOT_FROM_END:
  case R_RELAX_TLS_GD_TO_IE_END:
    return Body.getGotOffset() + A - In<ELFT>::Got->getSize();
  case R_GOT_OFF:
    return Body.getGotOffset() + A;
  case R_GOT_PAGE_PC:
  case R_RELAX_TLS_GD_TO_IE_PAGE_PC:
    return getAArch64Page(Body.getGotVA<ELFT>() + A) - getAArch64Page(P);
  case R_GOT_PC:
  case R_RELAX_TLS_GD_TO_IE:
    return Body.getGotVA<ELFT>() + A - P;
  case R_HINT:
  case R_NONE:
  case R_TLSDESC_CALL:
    llvm_unreachable("cannot relocate hint relocs");
  case R_MIPS_GOTREL:
    return Body.getVA(A) - In<ELFT>::MipsGot->getGp();
  case R_MIPS_GOT_GP:
    return In<ELFT>::MipsGot->getGp() + A;
  case R_MIPS_GOT_GP_PC: {
    // R_MIPS_LO16 expression has R_MIPS_GOT_GP_PC type iif the target
    // is _gp_disp symbol. In that case we should use the following
    // formula for calculation "AHL + GP - P + 4". For details see p. 4-19 at
    // ftp://www.linux-mips.org/pub/linux/mips/doc/ABI/mipsabi.pdf
    uint64_t V = In<ELFT>::MipsGot->getGp() + A - P;
    if (Type == R_MIPS_LO16)
      V += 4;
    return V;
  }
  case R_MIPS_GOT_LOCAL_PAGE:
    // If relocation against MIPS local symbol requires GOT entry, this entry
    // should be initialized by 'page address'. This address is high 16-bits
    // of sum the symbol's value and the addend.
    return In<ELFT>::MipsGot->getVA() +
           In<ELFT>::MipsGot->getPageEntryOffset(Body, A) -
           In<ELFT>::MipsGot->getGp();
  case R_MIPS_GOT_OFF:
  case R_MIPS_GOT_OFF32:
    // In case of MIPS if a GOT relocation has non-zero addend this addend
    // should be applied to the GOT entry content not to the GOT entry offset.
    // That is why we use separate expression type.
    return In<ELFT>::MipsGot->getVA() +
           In<ELFT>::MipsGot->getBodyEntryOffset(Body, A) -
           In<ELFT>::MipsGot->getGp();
  case R_MIPS_TLSGD:
    return In<ELFT>::MipsGot->getVA() + In<ELFT>::MipsGot->getTlsOffset() +
           In<ELFT>::MipsGot->getGlobalDynOffset(Body) -
           In<ELFT>::MipsGot->getGp();
  case R_MIPS_TLSLD:
    return In<ELFT>::MipsGot->getVA() + In<ELFT>::MipsGot->getTlsOffset() +
           In<ELFT>::MipsGot->getTlsIndexOff() - In<ELFT>::MipsGot->getGp();
  case R_PAGE_PC:
  case R_PLT_PAGE_PC:
    if (Body.isUndefined() && !Body.isLocal() && Body.symbol()->isWeak())
      return getAArch64Page(A);
    return getAArch64Page(Body.getVA(A)) - getAArch64Page(P);
  case R_PC:
    if (Body.isUndefined() && !Body.isLocal() && Body.symbol()->isWeak()) {
      // On ARM and AArch64 a branch to an undefined weak resolves to the
      // next instruction, otherwise the place.
      if (Config->EMachine == EM_ARM)
        return getARMUndefinedRelativeWeakVA(Type, A, P);
      if (Config->EMachine == EM_AARCH64)
        return getAArch64UndefinedRelativeWeakVA(Type, A, P);
    }
    return Body.getVA(A) - P;
  case R_PLT:
    return Body.getPltVA() + A;
  case R_PLT_PC:
  case R_PPC_PLT_OPD:
    return Body.getPltVA() + A - P;
  case R_PPC_OPD: {
    uint64_t SymVA = Body.getVA(A);
    // If we have an undefined weak symbol, we might get here with a symbol
    // address of zero. That could overflow, but the code must be unreachable,
    // so don't bother doing anything at all.
    if (!SymVA)
      return 0;
    if (Out::Opd) {
      // If this is a local call, and we currently have the address of a
      // function-descriptor, get the underlying code address instead.
      uint64_t OpdStart = Out::Opd->Addr;
      uint64_t OpdEnd = OpdStart + Out::Opd->Size;
      bool InOpd = OpdStart <= SymVA && SymVA < OpdEnd;
      if (InOpd)
        SymVA = read64be(&Out::OpdBuf[SymVA - OpdStart]);
    }
    return SymVA - P;
  }
  case R_PPC_TOC:
    return getPPC64TocBase() + A;
  case R_RELAX_GOT_PC:
    return Body.getVA(A) - P;
  case R_RELAX_TLS_GD_TO_LE:
  case R_RELAX_TLS_IE_TO_LE:
  case R_RELAX_TLS_LD_TO_LE:
  case R_TLS:
    // A weak undefined TLS symbol resolves to the base of the TLS
    // block, i.e. gets a value of zero. If we pass --gc-sections to
    // lld and .tbss is not referenced, it gets reclaimed and we don't
    // create a TLS program header. Therefore, we resolve this
    // statically to zero.
    if (Body.isTls() && (Body.isLazy() || Body.isUndefined()) &&
        Body.symbol()->isWeak())
      return 0;
    if (Target->TcbSize)
      return Body.getVA(A) + alignTo(Target->TcbSize, Out::TlsPhdr->p_align);
    return Body.getVA(A) - Out::TlsPhdr->p_memsz;
  case R_RELAX_TLS_GD_TO_LE_NEG:
  case R_NEG_TLS:
    return Out::TlsPhdr->p_memsz - Body.getVA(A);
  case R_SIZE:
    return Body.getSize<ELFT>() + A;
  case R_TLSDESC:
    return In<ELFT>::Got->getGlobalDynAddr(Body) + A;
  case R_TLSDESC_PAGE:
    return getAArch64Page(In<ELFT>::Got->getGlobalDynAddr(Body) + A) -
           getAArch64Page(P);
  case R_TLSGD:
    return In<ELFT>::Got->getGlobalDynOffset(Body) + A -
           In<ELFT>::Got->getSize();
  case R_TLSGD_PC:
    return In<ELFT>::Got->getGlobalDynAddr(Body) + A - P;
  case R_TLSLD:
    return In<ELFT>::Got->getTlsIndexOff() + A - In<ELFT>::Got->getSize();
  case R_TLSLD_PC:
    return In<ELFT>::Got->getTlsIndexVA() + A - P;
  }
  llvm_unreachable("Invalid expression");
}

// This function applies relocations to sections without SHF_ALLOC bit.
// Such sections are never mapped to memory at runtime. Debug sections are
// an example. Relocations in non-alloc sections are much easier to
// handle than in allocated sections because it will never need complex
// treatement such as GOT or PLT (because at runtime no one refers them).
// So, we handle relocations for non-alloc sections directly in this
// function as a performance optimization.
template <class ELFT, class RelTy>
void InputSection::relocateNonAlloc(uint8_t *Buf, ArrayRef<RelTy> Rels) {
  for (const RelTy &Rel : Rels) {
    uint32_t Type = Rel.getType(Config->IsMips64EL);
    uint64_t Offset = getOffset(Rel.r_offset);
    uint8_t *BufLoc = Buf + Offset;
    int64_t Addend = getAddend<ELFT>(Rel);
    if (!RelTy::IsRela)
      Addend += Target->getImplicitAddend(BufLoc, Type);

    SymbolBody &Sym = this->getFile<ELFT>()->getRelocTargetSym(Rel);
    RelExpr Expr = Target->getRelExpr(Type, Sym, BufLoc);
    if (Expr == R_NONE)
      continue;
    if (Expr != R_ABS) {
      error(this->getLocation<ELFT>(Offset) + ": has non-ABS reloc");
      return;
    }

    uint64_t AddrLoc = this->OutSec->Addr + Offset;
    uint64_t SymVA = 0;
    if (!Sym.isTls() || Out::TlsPhdr)
      SymVA = SignExtend64<sizeof(typename ELFT::uint) * 8>(
          getRelocTargetVA<ELFT>(Type, Addend, AddrLoc, Sym, R_ABS));
    Target->relocateOne(BufLoc, Type, SymVA);
  }
}

template <class ELFT> elf::ObjectFile<ELFT> *InputSectionBase::getFile() const {
  return cast_or_null<elf::ObjectFile<ELFT>>(File);
}

template <class ELFT>
void InputSectionBase::relocate(uint8_t *Buf, uint8_t *BufEnd) {
  // scanReloc function in Writer.cpp constructs Relocations
  // vector only for SHF_ALLOC'ed sections. For other sections,
  // we handle relocations directly here.
  auto *IS = dyn_cast<InputSection>(this);
  if (IS && !(IS->Flags & SHF_ALLOC)) {
    if (IS->AreRelocsRela)
      IS->relocateNonAlloc<ELFT>(Buf, IS->template relas<ELFT>());
    else
      IS->relocateNonAlloc<ELFT>(Buf, IS->template rels<ELFT>());
    return;
  }

  const unsigned Bits = sizeof(typename ELFT::uint) * 8;
  for (const Relocation &Rel : Relocations) {
    uint64_t Offset = getOffset(Rel.Offset);
    uint8_t *BufLoc = Buf + Offset;
    uint32_t Type = Rel.Type;

    uint64_t AddrLoc = getOutputSection()->Addr + Offset;
    RelExpr Expr = Rel.Expr;
    uint64_t TargetVA = SignExtend64<Bits>(
        getRelocTargetVA<ELFT>(Type, Rel.Addend, AddrLoc, *Rel.Sym, Expr));

    switch (Expr) {
    case R_RELAX_GOT_PC:
    case R_RELAX_GOT_PC_NOPIC:
      Target->relaxGot(BufLoc, TargetVA);
      break;
    case R_RELAX_TLS_IE_TO_LE:
      Target->relaxTlsIeToLe(BufLoc, Type, TargetVA);
      break;
    case R_RELAX_TLS_LD_TO_LE:
      Target->relaxTlsLdToLe(BufLoc, Type, TargetVA);
      break;
    case R_RELAX_TLS_GD_TO_LE:
    case R_RELAX_TLS_GD_TO_LE_NEG:
      Target->relaxTlsGdToLe(BufLoc, Type, TargetVA);
      break;
    case R_RELAX_TLS_GD_TO_IE:
    case R_RELAX_TLS_GD_TO_IE_ABS:
    case R_RELAX_TLS_GD_TO_IE_PAGE_PC:
    case R_RELAX_TLS_GD_TO_IE_END:
      Target->relaxTlsGdToIe(BufLoc, Type, TargetVA);
      break;
    case R_PPC_PLT_OPD:
      // Patch a nop (0x60000000) to a ld.
      if (BufLoc + 8 <= BufEnd && read32be(BufLoc + 4) == 0x60000000)
        write32be(BufLoc + 4, 0xe8410028); // ld %r2, 40(%r1)
    // fallthrough
    default:
      Target->relocateOne(BufLoc, Type, TargetVA);
      break;
    }
  }
}

template <class ELFT> void InputSection::writeTo(uint8_t *Buf) {
  if (this->Type == SHT_NOBITS)
    return;

  if (auto *S = dyn_cast<SyntheticSection>(this)) {
    S->writeTo(Buf + OutSecOff);
    return;
  }

  // If -r or --emit-relocs is given, then an InputSection
  // may be a relocation section.
  if (this->Type == SHT_RELA) {
    copyRelocations<ELFT>(Buf + OutSecOff,
                          this->template getDataAs<typename ELFT::Rela>());
    return;
  }
  if (this->Type == SHT_REL) {
    copyRelocations<ELFT>(Buf + OutSecOff,
                          this->template getDataAs<typename ELFT::Rel>());
    return;
  }

  // Copy section contents from source object file to output file
  // and then apply relocations.
  memcpy(Buf + OutSecOff, Data.data(), Data.size());
  uint8_t *BufEnd = Buf + OutSecOff + Data.size();
  this->relocate<ELFT>(Buf, BufEnd);
}

void InputSection::replace(InputSection *Other) {
  this->Alignment = std::max(this->Alignment, Other->Alignment);
  Other->Repl = this->Repl;
  Other->Live = false;
}

template <class ELFT>
EhInputSection::EhInputSection(elf::ObjectFile<ELFT> *F,
                               const typename ELFT::Shdr *Header,
                               StringRef Name)
    : InputSectionBase(F, Header, Name, InputSectionBase::EHFrame) {
  // Mark .eh_frame sections as live by default because there are
  // usually no relocations that point to .eh_frames. Otherwise,
  // the garbage collector would drop all .eh_frame sections.
  this->Live = true;
}

bool EhInputSection::classof(const SectionBase *S) {
  return S->kind() == InputSectionBase::EHFrame;
}

// Returns the index of the first relocation that points to a region between
// Begin and Begin+Size.
template <class IntTy, class RelTy>
static unsigned getReloc(IntTy Begin, IntTy Size, const ArrayRef<RelTy> &Rels,
                         unsigned &RelocI) {
  // Start search from RelocI for fast access. That works because the
  // relocations are sorted in .eh_frame.
  for (unsigned N = Rels.size(); RelocI < N; ++RelocI) {
    const RelTy &Rel = Rels[RelocI];
    if (Rel.r_offset < Begin)
      continue;

    if (Rel.r_offset < Begin + Size)
      return RelocI;
    return -1;
  }
  return -1;
}

// .eh_frame is a sequence of CIE or FDE records.
// This function splits an input section into records and returns them.
template <class ELFT> void EhInputSection::split() {
  // Early exit if already split.
  if (!this->Pieces.empty())
    return;

  if (this->NumRelocations) {
    if (this->AreRelocsRela)
      split<ELFT>(this->relas<ELFT>());
    else
      split<ELFT>(this->rels<ELFT>());
    return;
  }
  split<ELFT>(makeArrayRef<typename ELFT::Rela>(nullptr, nullptr));
}

template <class ELFT, class RelTy>
void EhInputSection::split(ArrayRef<RelTy> Rels) {
  ArrayRef<uint8_t> Data = this->Data;
  unsigned RelI = 0;
  for (size_t Off = 0, End = Data.size(); Off != End;) {
    size_t Size = readEhRecordSize<ELFT>(this, Off);
    this->Pieces.emplace_back(Off, this, Size, getReloc(Off, Size, Rels, RelI));
    // The empty record is the end marker.
    if (Size == 4)
      break;
    Off += Size;
  }
}

static size_t findNull(ArrayRef<uint8_t> A, size_t EntSize) {
  // Optimize the common case.
  StringRef S((const char *)A.data(), A.size());
  if (EntSize == 1)
    return S.find(0);

  for (unsigned I = 0, N = S.size(); I != N; I += EntSize) {
    const char *B = S.begin() + I;
    if (std::all_of(B, B + EntSize, [](char C) { return C == 0; }))
      return I;
  }
  return StringRef::npos;
}

// Split SHF_STRINGS section. Such section is a sequence of
// null-terminated strings.
void MergeInputSection::splitStrings(ArrayRef<uint8_t> Data, size_t EntSize) {
  size_t Off = 0;
  bool IsAlloc = this->Flags & SHF_ALLOC;
  while (!Data.empty()) {
    size_t End = findNull(Data, EntSize);
    if (End == StringRef::npos)
      fatal(toString(this) + ": string is not null terminated");
    size_t Size = End + EntSize;
    Pieces.emplace_back(Off, !IsAlloc);
    Hashes.push_back(hash_value(toStringRef(Data.slice(0, Size))));
    Data = Data.slice(Size);
    Off += Size;
  }
}

// Split non-SHF_STRINGS section. Such section is a sequence of
// fixed size records.
void MergeInputSection::splitNonStrings(ArrayRef<uint8_t> Data,
                                        size_t EntSize) {
  size_t Size = Data.size();
  assert((Size % EntSize) == 0);
  bool IsAlloc = this->Flags & SHF_ALLOC;
  for (unsigned I = 0, N = Size; I != N; I += EntSize) {
    Hashes.push_back(hash_value(toStringRef(Data.slice(I, EntSize))));
    Pieces.emplace_back(I, !IsAlloc);
  }
}

template <class ELFT>
MergeInputSection::MergeInputSection(elf::ObjectFile<ELFT> *F,
                                     const typename ELFT::Shdr *Header,
                                     StringRef Name)
    : InputSectionBase(F, Header, Name, InputSectionBase::Merge) {}

// This function is called after we obtain a complete list of input sections
// that need to be linked. This is responsible to split section contents
// into small chunks for further processing.
//
// Note that this function is called from parallel_for_each. This must be
// thread-safe (i.e. no memory allocation from the pools).
void MergeInputSection::splitIntoPieces() {
  ArrayRef<uint8_t> Data = this->Data;
  uint64_t EntSize = this->Entsize;
  if (this->Flags & SHF_STRINGS)
    splitStrings(Data, EntSize);
  else
    splitNonStrings(Data, EntSize);

  if (Config->GcSections && (this->Flags & SHF_ALLOC))
    for (uint64_t Off : LiveOffsets)
      this->getSectionPiece(Off)->Live = true;
}

bool MergeInputSection::classof(const SectionBase *S) {
  return S->kind() == InputSectionBase::Merge;
}

// Do binary search to get a section piece at a given input offset.
SectionPiece *MergeInputSection::getSectionPiece(uint64_t Offset) {
  auto *This = static_cast<const MergeInputSection *>(this);
  return const_cast<SectionPiece *>(This->getSectionPiece(Offset));
}

template <class It, class T, class Compare>
static It fastUpperBound(It First, It Last, const T &Value, Compare Comp) {
  size_t Size = std::distance(First, Last);
  assert(Size != 0);
  while (Size != 1) {
    size_t H = Size / 2;
    const It MI = First + H;
    Size -= H;
    First = Comp(Value, *MI) ? First : First + H;
  }
  return Comp(Value, *First) ? First : First + 1;
}

const SectionPiece *MergeInputSection::getSectionPiece(uint64_t Offset) const {
  uint64_t Size = this->Data.size();
  if (Offset >= Size)
    fatal(toString(this) + ": entry is past the end of the section");

  // Find the element this offset points to.
  auto I = fastUpperBound(
      Pieces.begin(), Pieces.end(), Offset,
      [](const uint64_t &A, const SectionPiece &B) { return A < B.InputOff; });
  --I;
  return &*I;
}

// Returns the offset in an output section for a given input offset.
// Because contents of a mergeable section is not contiguous in output,
// it is not just an addition to a base output offset.
uint64_t MergeInputSection::getOffset(uint64_t Offset) const {
  // Initialize OffsetMap lazily.
  std::call_once(InitOffsetMap, [&] {
    OffsetMap.reserve(Pieces.size());
    for (const SectionPiece &Piece : Pieces)
      OffsetMap[Piece.InputOff] = Piece.OutputOff;
  });

  // Find a string starting at a given offset.
  auto It = OffsetMap.find(Offset);
  if (It != OffsetMap.end())
    return It->second;

  if (!this->Live)
    return 0;

  // If Offset is not at beginning of a section piece, it is not in the map.
  // In that case we need to search from the original section piece vector.
  const SectionPiece &Piece = *this->getSectionPiece(Offset);
  if (!Piece.Live)
    return 0;

  uint64_t Addend = Offset - Piece.InputOff;
  return Piece.OutputOff + Addend;
}

template InputSection::InputSection(elf::ObjectFile<ELF32LE> *,
                                    const ELF32LE::Shdr *, StringRef);
template InputSection::InputSection(elf::ObjectFile<ELF32BE> *,
                                    const ELF32BE::Shdr *, StringRef);
template InputSection::InputSection(elf::ObjectFile<ELF64LE> *,
                                    const ELF64LE::Shdr *, StringRef);
template InputSection::InputSection(elf::ObjectFile<ELF64BE> *,
                                    const ELF64BE::Shdr *, StringRef);

template std::string InputSectionBase::getLocation<ELF32LE>(uint64_t);
template std::string InputSectionBase::getLocation<ELF32BE>(uint64_t);
template std::string InputSectionBase::getLocation<ELF64LE>(uint64_t);
template std::string InputSectionBase::getLocation<ELF64BE>(uint64_t);

template std::string InputSectionBase::getSrcMsg<ELF32LE>(uint64_t);
template std::string InputSectionBase::getSrcMsg<ELF32BE>(uint64_t);
template std::string InputSectionBase::getSrcMsg<ELF64LE>(uint64_t);
template std::string InputSectionBase::getSrcMsg<ELF64BE>(uint64_t);

template std::string InputSectionBase::getObjMsg<ELF32LE>(uint64_t);
template std::string InputSectionBase::getObjMsg<ELF32BE>(uint64_t);
template std::string InputSectionBase::getObjMsg<ELF64LE>(uint64_t);
template std::string InputSectionBase::getObjMsg<ELF64BE>(uint64_t);

template void InputSection::writeTo<ELF32LE>(uint8_t *);
template void InputSection::writeTo<ELF32BE>(uint8_t *);
template void InputSection::writeTo<ELF64LE>(uint8_t *);
template void InputSection::writeTo<ELF64BE>(uint8_t *);

template elf::ObjectFile<ELF32LE> *InputSectionBase::getFile<ELF32LE>() const;
template elf::ObjectFile<ELF32BE> *InputSectionBase::getFile<ELF32BE>() const;
template elf::ObjectFile<ELF64LE> *InputSectionBase::getFile<ELF64LE>() const;
template elf::ObjectFile<ELF64BE> *InputSectionBase::getFile<ELF64BE>() const;

template MergeInputSection::MergeInputSection(elf::ObjectFile<ELF32LE> *,
                                              const ELF32LE::Shdr *, StringRef);
template MergeInputSection::MergeInputSection(elf::ObjectFile<ELF32BE> *,
                                              const ELF32BE::Shdr *, StringRef);
template MergeInputSection::MergeInputSection(elf::ObjectFile<ELF64LE> *,
                                              const ELF64LE::Shdr *, StringRef);
template MergeInputSection::MergeInputSection(elf::ObjectFile<ELF64BE> *,
                                              const ELF64BE::Shdr *, StringRef);

template EhInputSection::EhInputSection(elf::ObjectFile<ELF32LE> *,
                                        const ELF32LE::Shdr *, StringRef);
template EhInputSection::EhInputSection(elf::ObjectFile<ELF32BE> *,
                                        const ELF32BE::Shdr *, StringRef);
template EhInputSection::EhInputSection(elf::ObjectFile<ELF64LE> *,
                                        const ELF64LE::Shdr *, StringRef);
template EhInputSection::EhInputSection(elf::ObjectFile<ELF64BE> *,
                                        const ELF64BE::Shdr *, StringRef);

template void EhInputSection::split<ELF32LE>();
template void EhInputSection::split<ELF32BE>();
template void EhInputSection::split<ELF64LE>();
template void EhInputSection::split<ELF64BE>();
