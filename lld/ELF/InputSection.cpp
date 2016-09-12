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
#include "OutputSections.h"
#include "Target.h"
#include "Thunks.h"

#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

template <class ELFT>
static ArrayRef<uint8_t> getSectionContents(elf::ObjectFile<ELFT> *File,
                                            const typename ELFT::Shdr *Hdr) {
  if (!File || Hdr->sh_type == SHT_NOBITS)
    return {};
  return check(File->getObj().getSectionContents(Hdr));
}

template <class ELFT>
InputSectionBase<ELFT>::InputSectionBase(elf::ObjectFile<ELFT> *File,
                                         const Elf_Shdr *Hdr, StringRef Name,
                                         Kind SectionKind)
    : InputSectionData(SectionKind, Name, getSectionContents(File, Hdr),
                       Hdr->sh_flags & SHF_COMPRESSED, !Config->GcSections),
      Header(Hdr), File(File), Repl(this) {
  // The ELF spec states that a value of 0 means the section has
  // no alignment constraits.
  Alignment = std::max<uintX_t>(Header->sh_addralign, 1);
}

template <class ELFT> size_t InputSectionBase<ELFT>::getSize() const {
  if (auto *D = dyn_cast<InputSection<ELFT>>(this))
    if (D->getThunksSize() > 0)
      return D->getThunkOff() + D->getThunksSize();
  return Header->sh_size;
}

// Returns a string for an error message.
template <class SectionT> static std::string getName(SectionT *Sec) {
  return (Sec->getFile()->getName() + "(" + Sec->Name + ")").str();
}

template <class ELFT>
typename ELFT::uint InputSectionBase<ELFT>::getOffset(uintX_t Offset) const {
  switch (kind()) {
  case Regular:
    return cast<InputSection<ELFT>>(this)->OutSecOff + Offset;
  case EHFrame:
    // The file crtbeginT.o has relocations pointing to the start of an empty
    // .eh_frame that is known to be the first in the link. It does that to
    // identify the start of the output .eh_frame.
    return Offset;
  case Merge:
    return cast<MergeInputSection<ELFT>>(this)->getOffset(Offset);
  case MipsReginfo:
  case MipsOptions:
  case MipsAbiFlags:
    // MIPS .reginfo, .MIPS.options, and .MIPS.abiflags sections are consumed
    // by the linker, and the linker produces a single output section. It is
    // possible that input files contain section symbol points to the
    // corresponding input section. Redirect it to the produced output section.
    if (Offset != 0)
      fatal(getName(this) + ": unsupported reference to the middle of '" +
            Name + "' section");
    return this->OutSec->getVA();
  }
  llvm_unreachable("invalid section kind");
}

template <class ELFT> void InputSectionBase<ELFT>::uncompress() {
  if (!zlib::isAvailable())
    fatal(getName(this) +
          ": build lld with zlib to enable compressed sections support");

  // A compressed section consists of a header of Elf_Chdr type
  // followed by compressed data.
  if (Data.size() < sizeof(Elf_Chdr))
    fatal("corrupt compressed section");

  auto *Hdr = reinterpret_cast<const Elf_Chdr *>(Data.data());
  Data = Data.slice(sizeof(Elf_Chdr));

  if (Hdr->ch_type != ELFCOMPRESS_ZLIB)
    fatal(getName(this) + ": unsupported compression type");

  StringRef Buf((const char *)Data.data(), Data.size());
  size_t UncompressedDataSize = Hdr->ch_size;
  UncompressedData.reset(new char[UncompressedDataSize]);
  if (zlib::uncompress(Buf, UncompressedData.get(), UncompressedDataSize) !=
      zlib::StatusOK)
    fatal(getName(this) + ": error uncompressing section");
  Data = ArrayRef<uint8_t>((uint8_t *)UncompressedData.get(),
                           UncompressedDataSize);
}

template <class ELFT>
typename ELFT::uint
InputSectionBase<ELFT>::getOffset(const DefinedRegular<ELFT> &Sym) const {
  return getOffset(Sym.Value);
}

template <class ELFT>
InputSection<ELFT>::InputSection(elf::ObjectFile<ELFT> *F,
                                 const Elf_Shdr *Header, StringRef Name)
    : InputSectionBase<ELFT>(F, Header, Name, Base::Regular) {}

template <class ELFT>
bool InputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == Base::Regular;
}

template <class ELFT>
InputSectionBase<ELFT> *InputSection<ELFT>::getRelocatedSection() {
  assert(this->Header->sh_type == SHT_RELA || this->Header->sh_type == SHT_REL);
  ArrayRef<InputSectionBase<ELFT> *> Sections = this->File->getSections();
  return Sections[this->Header->sh_info];
}

template <class ELFT>
void InputSection<ELFT>::addThunk(const Thunk<ELFT> *T) {
  Thunks.push_back(T);
}

template <class ELFT> uint64_t InputSection<ELFT>::getThunkOff() const {
  return this->Header->sh_size;
}

template <class ELFT> uint64_t InputSection<ELFT>::getThunksSize() const {
  uint64_t Total = 0;
  for (const Thunk<ELFT> *T : Thunks)
    Total += T->size();
  return Total;
}

// This is used for -r. We can't use memcpy to copy relocations because we need
// to update symbol table offset and section index for each relocation. So we
// copy relocations one by one.
template <class ELFT>
template <class RelTy>
void InputSection<ELFT>::copyRelocations(uint8_t *Buf, ArrayRef<RelTy> Rels) {
  InputSectionBase<ELFT> *RelocatedSection = getRelocatedSection();

  for (const RelTy &Rel : Rels) {
    uint32_t Type = Rel.getType(Config->Mips64EL);
    SymbolBody &Body = this->File->getRelocTargetSym(Rel);

    Elf_Rela *P = reinterpret_cast<Elf_Rela *>(Buf);
    Buf += sizeof(RelTy);

    if (Config->Rela)
      P->r_addend = getAddend<ELFT>(Rel);
    P->r_offset = RelocatedSection->getOffset(Rel.r_offset);
    P->setSymbolAndType(Body.DynsymIndex, Type, Config->Mips64EL);
  }
}

// Page(Expr) is the page address of the expression Expr, defined
// as (Expr & ~0xFFF). (This applies even if the machine page size
// supported by the platform has a different value.)
static uint64_t getAArch64Page(uint64_t Expr) {
  return Expr & (~static_cast<uint64_t>(0xFFF));
}

template <class ELFT>
static typename ELFT::uint getSymVA(uint32_t Type, typename ELFT::uint A,
                                    typename ELFT::uint P,
                                    const SymbolBody &Body, RelExpr Expr) {
  switch (Expr) {
  case R_HINT:
    llvm_unreachable("cannot relocate hint relocs");
  case R_TLSLD:
    return Out<ELFT>::Got->getTlsIndexOff() + A - Out<ELFT>::Got->getSize();
  case R_TLSLD_PC:
    return Out<ELFT>::Got->getTlsIndexVA() + A - P;
  case R_THUNK_ABS:
    return Body.getThunkVA<ELFT>() + A;
  case R_THUNK_PC:
  case R_THUNK_PLT_PC:
    return Body.getThunkVA<ELFT>() + A - P;
  case R_PPC_TOC:
    return getPPC64TocBase() + A;
  case R_TLSGD:
    return Out<ELFT>::Got->getGlobalDynOffset(Body) + A -
           Out<ELFT>::Got->getSize();
  case R_TLSGD_PC:
    return Out<ELFT>::Got->getGlobalDynAddr(Body) + A - P;
  case R_TLSDESC:
    return Out<ELFT>::Got->getGlobalDynAddr(Body) + A;
  case R_TLSDESC_PAGE:
    return getAArch64Page(Out<ELFT>::Got->getGlobalDynAddr(Body) + A) -
           getAArch64Page(P);
  case R_PLT:
    return Body.getPltVA<ELFT>() + A;
  case R_PLT_PC:
  case R_PPC_PLT_OPD:
    return Body.getPltVA<ELFT>() + A - P;
  case R_SIZE:
    return Body.getSize<ELFT>() + A;
  case R_GOTREL:
    return Body.getVA<ELFT>(A) - Out<ELFT>::Got->getVA();
  case R_GOTREL_FROM_END:
    return Body.getVA<ELFT>(A) - Out<ELFT>::Got->getVA() -
           Out<ELFT>::Got->getSize();
  case R_RELAX_TLS_GD_TO_IE_END:
  case R_GOT_FROM_END:
    return Body.getGotOffset<ELFT>() + A - Out<ELFT>::Got->getSize();
  case R_RELAX_TLS_GD_TO_IE_ABS:
  case R_GOT:
    return Body.getGotVA<ELFT>() + A;
  case R_RELAX_TLS_GD_TO_IE_PAGE_PC:
  case R_GOT_PAGE_PC:
    return getAArch64Page(Body.getGotVA<ELFT>() + A) - getAArch64Page(P);
  case R_RELAX_TLS_GD_TO_IE:
  case R_GOT_PC:
    return Body.getGotVA<ELFT>() + A - P;
  case R_GOTONLY_PC:
    return Out<ELFT>::Got->getVA() + A - P;
  case R_GOTONLY_PC_FROM_END:
    return Out<ELFT>::Got->getVA() + A - P + Out<ELFT>::Got->getSize();
  case R_RELAX_TLS_LD_TO_LE:
  case R_RELAX_TLS_IE_TO_LE:
  case R_RELAX_TLS_GD_TO_LE:
  case R_TLS:
    if (Target->TcbSize)
      return Body.getVA<ELFT>(A) +
             alignTo(Target->TcbSize, Out<ELFT>::TlsPhdr->p_align);
    return Body.getVA<ELFT>(A) - Out<ELFT>::TlsPhdr->p_memsz;
  case R_RELAX_TLS_GD_TO_LE_NEG:
  case R_NEG_TLS:
    return Out<ELF32LE>::TlsPhdr->p_memsz - Body.getVA<ELFT>(A);
  case R_ABS:
  case R_RELAX_GOT_PC_NOPIC:
    return Body.getVA<ELFT>(A);
  case R_GOT_OFF:
    return Body.getGotOffset<ELFT>() + A;
  case R_MIPS_GOT_LOCAL_PAGE:
    // If relocation against MIPS local symbol requires GOT entry, this entry
    // should be initialized by 'page address'. This address is high 16-bits
    // of sum the symbol's value and the addend.
    return Out<ELFT>::Got->getMipsLocalPageOffset(Body.getVA<ELFT>(A));
  case R_MIPS_GOT_OFF:
    // In case of MIPS if a GOT relocation has non-zero addend this addend
    // should be applied to the GOT entry content not to the GOT entry offset.
    // That is why we use separate expression type.
    return Out<ELFT>::Got->getMipsGotOffset(Body, A);
  case R_MIPS_TLSGD:
    return Out<ELFT>::Got->getGlobalDynOffset(Body) +
           Out<ELFT>::Got->getMipsTlsOffset() - MipsGPOffset;
  case R_MIPS_TLSLD:
    return Out<ELFT>::Got->getTlsIndexOff() +
           Out<ELFT>::Got->getMipsTlsOffset() - MipsGPOffset;
  case R_PPC_OPD: {
    uint64_t SymVA = Body.getVA<ELFT>(A);
    // If we have an undefined weak symbol, we might get here with a symbol
    // address of zero. That could overflow, but the code must be unreachable,
    // so don't bother doing anything at all.
    if (!SymVA)
      return 0;
    if (Out<ELF64BE>::Opd) {
      // If this is a local call, and we currently have the address of a
      // function-descriptor, get the underlying code address instead.
      uint64_t OpdStart = Out<ELF64BE>::Opd->getVA();
      uint64_t OpdEnd = OpdStart + Out<ELF64BE>::Opd->getSize();
      bool InOpd = OpdStart <= SymVA && SymVA < OpdEnd;
      if (InOpd)
        SymVA = read64be(&Out<ELF64BE>::OpdBuf[SymVA - OpdStart]);
    }
    return SymVA - P;
  }
  case R_PC:
  case R_RELAX_GOT_PC:
    return Body.getVA<ELFT>(A) - P;
  case R_PLT_PAGE_PC:
  case R_PAGE_PC:
    return getAArch64Page(Body.getVA<ELFT>(A)) - getAArch64Page(P);
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
template <class ELFT>
template <class RelTy>
void InputSection<ELFT>::relocateNonAlloc(uint8_t *Buf, ArrayRef<RelTy> Rels) {
  const unsigned Bits = sizeof(uintX_t) * 8;
  for (const RelTy &Rel : Rels) {
    uint32_t Type = Rel.getType(Config->Mips64EL);
    uintX_t Offset = this->getOffset(Rel.r_offset);
    uint8_t *BufLoc = Buf + Offset;
    uintX_t Addend = getAddend<ELFT>(Rel);
    if (!RelTy::IsRela)
      Addend += Target->getImplicitAddend(BufLoc, Type);

    SymbolBody &Sym = this->File->getRelocTargetSym(Rel);
    if (Target->getRelExpr(Type, Sym) != R_ABS) {
      error(getName(this) + " has non-ABS reloc");
      return;
    }

    uintX_t AddrLoc = this->OutSec->getVA() + Offset;
    uint64_t SymVA =
        SignExtend64<Bits>(getSymVA<ELFT>(Type, Addend, AddrLoc, Sym, R_ABS));
    Target->relocateOne(BufLoc, Type, SymVA);
  }
}

template <class ELFT>
void InputSectionBase<ELFT>::relocate(uint8_t *Buf, uint8_t *BufEnd) {
  // scanReloc function in Writer.cpp constructs Relocations
  // vector only for SHF_ALLOC'ed sections. For other sections,
  // we handle relocations directly here.
  auto *IS = dyn_cast<InputSection<ELFT>>(this);
  if (IS && !(IS->Header->sh_flags & SHF_ALLOC)) {
    for (const Elf_Shdr *RelSec : IS->RelocSections) {
      if (RelSec->sh_type == SHT_RELA)
        IS->relocateNonAlloc(Buf, IS->File->getObj().relas(RelSec));
      else
        IS->relocateNonAlloc(Buf, IS->File->getObj().rels(RelSec));
    }
    return;
  }

  const unsigned Bits = sizeof(uintX_t) * 8;
  for (const Relocation &Rel : Relocations) {
    uintX_t Offset = getOffset(Rel.Offset);
    uint8_t *BufLoc = Buf + Offset;
    uint32_t Type = Rel.Type;
    uintX_t A = Rel.Addend;

    uintX_t AddrLoc = OutSec->getVA() + Offset;
    RelExpr Expr = Rel.Expr;
    uint64_t SymVA =
        SignExtend64<Bits>(getSymVA<ELFT>(Type, A, AddrLoc, *Rel.Sym, Expr));

    switch (Expr) {
    case R_RELAX_GOT_PC:
    case R_RELAX_GOT_PC_NOPIC:
      Target->relaxGot(BufLoc, SymVA);
      break;
    case R_RELAX_TLS_IE_TO_LE:
      Target->relaxTlsIeToLe(BufLoc, Type, SymVA);
      break;
    case R_RELAX_TLS_LD_TO_LE:
      Target->relaxTlsLdToLe(BufLoc, Type, SymVA);
      break;
    case R_RELAX_TLS_GD_TO_LE:
    case R_RELAX_TLS_GD_TO_LE_NEG:
      Target->relaxTlsGdToLe(BufLoc, Type, SymVA);
      break;
    case R_RELAX_TLS_GD_TO_IE:
    case R_RELAX_TLS_GD_TO_IE_ABS:
    case R_RELAX_TLS_GD_TO_IE_PAGE_PC:
    case R_RELAX_TLS_GD_TO_IE_END:
      Target->relaxTlsGdToIe(BufLoc, Type, SymVA);
      break;
    case R_PPC_PLT_OPD:
      // Patch a nop (0x60000000) to a ld.
      if (BufLoc + 8 <= BufEnd && read32be(BufLoc + 4) == 0x60000000)
        write32be(BufLoc + 4, 0xe8410028); // ld %r2, 40(%r1)
      // fallthrough
    default:
      Target->relocateOne(BufLoc, Type, SymVA);
      break;
    }
  }
}

template <class ELFT> void InputSection<ELFT>::writeTo(uint8_t *Buf) {
  if (this->Header->sh_type == SHT_NOBITS)
    return;
  ELFFile<ELFT> &EObj = this->File->getObj();

  // If -r is given, then an InputSection may be a relocation section.
  if (this->Header->sh_type == SHT_RELA) {
    copyRelocations(Buf + OutSecOff, EObj.relas(this->Header));
    return;
  }
  if (this->Header->sh_type == SHT_REL) {
    copyRelocations(Buf + OutSecOff, EObj.rels(this->Header));
    return;
  }

  // Copy section contents from source object file to output file.
  ArrayRef<uint8_t> Data = this->Data;
  memcpy(Buf + OutSecOff, Data.data(), Data.size());

  // Iterate over all relocation sections that apply to this section.
  uint8_t *BufEnd = Buf + OutSecOff + Data.size();
  this->relocate(Buf, BufEnd);

  // The section might have a data/code generated by the linker and need
  // to be written after the section. Usually these are thunks - small piece
  // of code used to jump between "incompatible" functions like PIC and non-PIC
  // or if the jump target too far and its address does not fit to the short
  // jump istruction.
  if (!Thunks.empty()) {
    Buf += OutSecOff + getThunkOff();
    for (const Thunk<ELFT> *T : Thunks) {
      T->writeTo(Buf);
      Buf += T->size();
    }
  }
}

template <class ELFT>
void InputSection<ELFT>::replace(InputSection<ELFT> *Other) {
  this->Alignment = std::max(this->Alignment, Other->Alignment);
  Other->Repl = this->Repl;
  Other->Live = false;
}

template <class ELFT>
EhInputSection<ELFT>::EhInputSection(elf::ObjectFile<ELFT> *F,
                                     const Elf_Shdr *Header, StringRef Name)
    : InputSectionBase<ELFT>(F, Header, Name, InputSectionBase<ELFT>::EHFrame) {
  // Mark .eh_frame sections as live by default because there are
  // usually no relocations that point to .eh_frames. Otherwise,
  // the garbage collector would drop all .eh_frame sections.
  this->Live = true;
}

template <class ELFT>
bool EhInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == InputSectionBase<ELFT>::EHFrame;
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
template <class ELFT>
void EhInputSection<ELFT>::split() {
  // Early exit if already split.
  if (!this->Pieces.empty())
    return;

  if (RelocSection) {
    ELFFile<ELFT> &Obj = this->File->getObj();
    if (RelocSection->sh_type == SHT_RELA)
      split(Obj.relas(RelocSection));
    else
      split(Obj.rels(RelocSection));
    return;
  }
  split(makeArrayRef<typename ELFT::Rela>(nullptr, nullptr));
}

template <class ELFT>
template <class RelTy>
void EhInputSection<ELFT>::split(ArrayRef<RelTy> Rels) {
  ArrayRef<uint8_t> Data = this->Data;
  unsigned RelI = 0;
  for (size_t Off = 0, End = Data.size(); Off != End;) {
    size_t Size = readEhRecordSize<ELFT>(Data.slice(Off));
    this->Pieces.emplace_back(Off, Data.slice(Off, Size),
                              getReloc(Off, Size, Rels, RelI));
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
template <class ELFT>
std::vector<SectionPiece>
MergeInputSection<ELFT>::splitStrings(ArrayRef<uint8_t> Data, size_t EntSize) {
  std::vector<SectionPiece> V;
  size_t Off = 0;
  while (!Data.empty()) {
    size_t End = findNull(Data, EntSize);
    if (End == StringRef::npos)
      fatal(getName(this) + ": string is not null terminated");
    size_t Size = End + EntSize;
    V.emplace_back(Off, Data.slice(0, Size));
    Data = Data.slice(Size);
    Off += Size;
  }
  return V;
}

// Split non-SHF_STRINGS section. Such section is a sequence of
// fixed size records.
template <class ELFT>
std::vector<SectionPiece>
MergeInputSection<ELFT>::splitNonStrings(ArrayRef<uint8_t> Data,
                                         size_t EntSize) {
  std::vector<SectionPiece> V;
  size_t Size = Data.size();
  assert((Size % EntSize) == 0);
  for (unsigned I = 0, N = Size; I != N; I += EntSize)
    V.emplace_back(I, Data.slice(I, EntSize));
  return V;
}

template <class ELFT>
MergeInputSection<ELFT>::MergeInputSection(elf::ObjectFile<ELFT> *F,
                                           const Elf_Shdr *Header,
                                           StringRef Name)
    : InputSectionBase<ELFT>(F, Header, Name, InputSectionBase<ELFT>::Merge) {}

template <class ELFT> void MergeInputSection<ELFT>::splitIntoPieces() {
  ArrayRef<uint8_t> Data = this->Data;
  uintX_t EntSize = this->Header->sh_entsize;
  if (this->Header->sh_flags & SHF_STRINGS)
    this->Pieces = splitStrings(Data, EntSize);
  else
    this->Pieces = splitNonStrings(Data, EntSize);

  if (Config->GcSections)
    for (uintX_t Off : LiveOffsets)
      this->getSectionPiece(Off)->Live = true;
}

template <class ELFT>
bool MergeInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == InputSectionBase<ELFT>::Merge;
}

// Do binary search to get a section piece at a given input offset.
template <class ELFT>
SectionPiece *MergeInputSection<ELFT>::getSectionPiece(uintX_t Offset) {
  auto *This = static_cast<const MergeInputSection<ELFT> *>(this);
  return const_cast<SectionPiece *>(This->getSectionPiece(Offset));
}

template <class ELFT>
const SectionPiece *
MergeInputSection<ELFT>::getSectionPiece(uintX_t Offset) const {
  uintX_t Size = this->Data.size();
  if (Offset >= Size)
    fatal(getName(this) + ": entry is past the end of the section");

  // Find the element this offset points to.
  auto I = std::upper_bound(
      Pieces.begin(), Pieces.end(), Offset,
      [](const uintX_t &A, const SectionPiece &B) { return A < B.InputOff; });
  --I;
  return &*I;
}

// Returns the offset in an output section for a given input offset.
// Because contents of a mergeable section is not contiguous in output,
// it is not just an addition to a base output offset.
template <class ELFT>
typename ELFT::uint MergeInputSection<ELFT>::getOffset(uintX_t Offset) const {
  auto It = OffsetMap.find(Offset);
  if (It != OffsetMap.end())
    return It->second;

  // If Offset is not at beginning of a section piece, it is not in the map.
  // In that case we need to search from the original section piece vector.
  const SectionPiece &Piece = *this->getSectionPiece(Offset);
  assert(Piece.Live);
  uintX_t Addend = Offset - Piece.InputOff;
  return Piece.OutputOff + Addend;
}

// Create a map from input offsets to output offsets for all section pieces.
// It is called after finalize().
template <class ELFT> void  MergeInputSection<ELFT>::finalizePieces() {
  OffsetMap.grow(this->Pieces.size());
  for (SectionPiece &Piece : this->Pieces) {
    if (!Piece.Live)
      continue;
    if (Piece.OutputOff == size_t(-1)) {
      // Offsets of tail-merged strings are computed lazily.
      auto *OutSec = static_cast<MergeOutputSection<ELFT> *>(this->OutSec);
      ArrayRef<uint8_t> D = Piece.data();
      StringRef S((const char *)D.data(), D.size());
      Piece.OutputOff = OutSec->getOffset(S);
    }
    OffsetMap[Piece.InputOff] = Piece.OutputOff;
  }
}

template <class ELFT>
MipsReginfoInputSection<ELFT>::MipsReginfoInputSection(elf::ObjectFile<ELFT> *F,
                                                       const Elf_Shdr *Hdr,
                                                       StringRef Name)
    : InputSectionBase<ELFT>(F, Hdr, Name,
                             InputSectionBase<ELFT>::MipsReginfo) {
  ArrayRef<uint8_t> Data = this->Data;
  // Initialize this->Reginfo.
  if (Data.size() != sizeof(Elf_Mips_RegInfo<ELFT>)) {
    error(getName(this) + ": invalid size of .reginfo section");
    return;
  }
  Reginfo = reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(Data.data());
}

template <class ELFT>
bool MipsReginfoInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == InputSectionBase<ELFT>::MipsReginfo;
}

template <class ELFT>
MipsOptionsInputSection<ELFT>::MipsOptionsInputSection(elf::ObjectFile<ELFT> *F,
                                                       const Elf_Shdr *Hdr,
                                                       StringRef Name)
    : InputSectionBase<ELFT>(F, Hdr, Name,
                             InputSectionBase<ELFT>::MipsOptions) {
  // Find ODK_REGINFO option in the section's content.
  ArrayRef<uint8_t> D = this->Data;
  while (!D.empty()) {
    if (D.size() < sizeof(Elf_Mips_Options<ELFT>)) {
      error(getName(this) + ": invalid size of .MIPS.options section");
      break;
    }
    auto *O = reinterpret_cast<const Elf_Mips_Options<ELFT> *>(D.data());
    if (O->kind == ODK_REGINFO) {
      Reginfo = &O->getRegInfo();
      break;
    }
    D = D.slice(O->size);
  }
}

template <class ELFT>
bool MipsOptionsInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == InputSectionBase<ELFT>::MipsOptions;
}

template <class ELFT>
MipsAbiFlagsInputSection<ELFT>::MipsAbiFlagsInputSection(
    elf::ObjectFile<ELFT> *F, const Elf_Shdr *Hdr, StringRef Name)
    : InputSectionBase<ELFT>(F, Hdr, Name,
                             InputSectionBase<ELFT>::MipsAbiFlags) {
  // Initialize this->Flags.
  ArrayRef<uint8_t> Data = this->Data;
  if (Data.size() != sizeof(Elf_Mips_ABIFlags<ELFT>)) {
    error("invalid size of .MIPS.abiflags section");
    return;
  }
  Flags = reinterpret_cast<const Elf_Mips_ABIFlags<ELFT> *>(Data.data());
}

template <class ELFT>
bool MipsAbiFlagsInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->kind() == InputSectionBase<ELFT>::MipsAbiFlags;
}

template <class ELFT>
CommonInputSection<ELFT>::CommonInputSection(std::vector<DefinedCommon *> Syms)
    : InputSection<ELFT>(nullptr, &Hdr, "") {
  Hdr.sh_size = 0;
  Hdr.sh_type = SHT_NOBITS;
  Hdr.sh_flags = SHF_ALLOC | SHF_WRITE;
  this->Live = true;

  // Sort the common symbols by alignment as an heuristic to pack them better.
  std::stable_sort(Syms.begin(), Syms.end(),
                   [](const DefinedCommon *A, const DefinedCommon *B) {
                     return A->Alignment > B->Alignment;
                   });

  for (DefinedCommon *Sym : Syms) {
    this->Alignment = std::max<uintX_t>(this->Alignment, Sym->Alignment);
    Hdr.sh_size = alignTo(Hdr.sh_size, Sym->Alignment);

    // Compute symbol offset relative to beginning of input section.
    Sym->Offset = Hdr.sh_size;
    Hdr.sh_size += Sym->Size;
  }
}

template class elf::InputSectionBase<ELF32LE>;
template class elf::InputSectionBase<ELF32BE>;
template class elf::InputSectionBase<ELF64LE>;
template class elf::InputSectionBase<ELF64BE>;

template class elf::InputSection<ELF32LE>;
template class elf::InputSection<ELF32BE>;
template class elf::InputSection<ELF64LE>;
template class elf::InputSection<ELF64BE>;

template class elf::EhInputSection<ELF32LE>;
template class elf::EhInputSection<ELF32BE>;
template class elf::EhInputSection<ELF64LE>;
template class elf::EhInputSection<ELF64BE>;

template class elf::MergeInputSection<ELF32LE>;
template class elf::MergeInputSection<ELF32BE>;
template class elf::MergeInputSection<ELF64LE>;
template class elf::MergeInputSection<ELF64BE>;

template class elf::MipsReginfoInputSection<ELF32LE>;
template class elf::MipsReginfoInputSection<ELF32BE>;
template class elf::MipsReginfoInputSection<ELF64LE>;
template class elf::MipsReginfoInputSection<ELF64BE>;

template class elf::MipsOptionsInputSection<ELF32LE>;
template class elf::MipsOptionsInputSection<ELF32BE>;
template class elf::MipsOptionsInputSection<ELF64LE>;
template class elf::MipsOptionsInputSection<ELF64BE>;

template class elf::MipsAbiFlagsInputSection<ELF32LE>;
template class elf::MipsAbiFlagsInputSection<ELF32BE>;
template class elf::MipsAbiFlagsInputSection<ELF64LE>;
template class elf::MipsAbiFlagsInputSection<ELF64BE>;

template class elf::CommonInputSection<ELF32LE>;
template class elf::CommonInputSection<ELF32BE>;
template class elf::CommonInputSection<ELF64LE>;
template class elf::CommonInputSection<ELF64BE>;
