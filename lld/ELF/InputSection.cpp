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
#include "Error.h"
#include "InputFiles.h"
#include "OutputSections.h"
#include "Target.h"

#include "llvm/Support/Endian.h"

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

template <class ELFT>
InputSectionBase<ELFT>::InputSectionBase(elf::ObjectFile<ELFT> *File,
                                         const Elf_Shdr *Header,
                                         Kind SectionKind)
    : Header(Header), File(File), SectionKind(SectionKind), Repl(this) {
  // The garbage collector sets sections' Live bits.
  // If GC is disabled, all sections are considered live by default.
  Live = !Config->GcSections;

  // The ELF spec states that a value of 0 means the section has
  // no alignment constraits.
  Align = std::max<uintX_t>(Header->sh_addralign, 1);
}

template <class ELFT> size_t InputSectionBase<ELFT>::getSize() const {
  if (auto *D = dyn_cast<InputSection<ELFT>>(this))
    if (D->getThunksSize() > 0)
      return D->getThunkOff() + D->getThunksSize();
  return Header->sh_size;
}

template <class ELFT> StringRef InputSectionBase<ELFT>::getSectionName() const {
  return check(File->getObj().getSectionName(this->Header));
}

template <class ELFT>
ArrayRef<uint8_t> InputSectionBase<ELFT>::getSectionData() const {
  return check(this->File->getObj().getSectionContents(this->Header));
}

template <class ELFT>
typename ELFT::uint InputSectionBase<ELFT>::getOffset(uintX_t Offset) {
  switch (SectionKind) {
  case Regular:
    return cast<InputSection<ELFT>>(this)->OutSecOff + Offset;
  case EHFrame:
    return cast<EHInputSection<ELFT>>(this)->getOffset(Offset);
  case Merge:
    return cast<MergeInputSection<ELFT>>(this)->getOffset(Offset);
  case MipsReginfo:
    // MIPS .reginfo sections are consumed by the linker,
    // so it should never be copied to output.
    llvm_unreachable("MIPS .reginfo reached writeTo().");
  }
  llvm_unreachable("invalid section kind");
}

template <class ELFT>
typename ELFT::uint
InputSectionBase<ELFT>::getOffset(const DefinedRegular<ELFT> &Sym) {
  return getOffset(Sym.Value);
}

template <class ELFT>
InputSection<ELFT>::InputSection(elf::ObjectFile<ELFT> *F,
                                 const Elf_Shdr *Header)
    : InputSectionBase<ELFT>(F, Header, Base::Regular) {}

template <class ELFT>
bool InputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == Base::Regular;
}

template <class ELFT>
InputSectionBase<ELFT> *InputSection<ELFT>::getRelocatedSection() {
  assert(this->Header->sh_type == SHT_RELA || this->Header->sh_type == SHT_REL);
  ArrayRef<InputSectionBase<ELFT> *> Sections = this->File->getSections();
  return Sections[this->Header->sh_info];
}

template <class ELFT> void InputSection<ELFT>::addThunk(SymbolBody &Body) {
  Body.ThunkIndex = Thunks.size();
  Thunks.push_back(&Body);
}

template <class ELFT> uint64_t InputSection<ELFT>::getThunkOff() const {
  return this->Header->sh_size;
}

template <class ELFT> uint64_t InputSection<ELFT>::getThunksSize() const {
  return Thunks.size() * Target->ThunkSize;
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

    RelTy *P = reinterpret_cast<RelTy *>(Buf);
    Buf += sizeof(RelTy);

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
static typename ELFT::uint
getSymVA(uint32_t Type, typename ELFT::uint A, typename ELFT::uint P,
         const SymbolBody &Body, uint8_t *BufLoc,
         const elf::ObjectFile<ELFT> &File, RelExpr Expr) {
  typedef typename ELFT::uint uintX_t;
  switch (Expr) {
  case R_TLSLD:
    return Out<ELFT>::Got->getTlsIndexOff() + A -
           Out<ELFT>::Got->getNumEntries() * sizeof(uintX_t);
  case R_TLSLD_PC:
    return Out<ELFT>::Got->getTlsIndexVA() + A - P;
  case R_THUNK:
    return Body.getThunkVA<ELFT>();
  case R_PPC_TOC:
    return getPPC64TocBase() + A;
  case R_TLSGD:
    return Out<ELFT>::Got->getGlobalDynOffset(Body) + A -
           Out<ELFT>::Got->getNumEntries() * sizeof(uintX_t);
  case R_TLSGD_PC:
    return Out<ELFT>::Got->getGlobalDynAddr(Body) + A - P;
  case R_PLT:
    return Body.getPltVA<ELFT>() + A;
  case R_PLT_PC:
  case R_PPC_PLT_OPD:
    return Body.getPltVA<ELFT>() + A - P;
  case R_SIZE:
    return Body.getSize<ELFT>() + A;
  case R_GOTREL:
    return Body.getVA<ELFT>(A) - Out<ELFT>::Got->getVA();
  case R_GOT_FROM_END:
    return Body.getGotOffset<ELFT>() + A -
           Out<ELFT>::Got->getNumEntries() * sizeof(uintX_t);
  case R_GOT:
  case R_RELAX_TLS_GD_TO_IE:
    return Body.getGotVA<ELFT>() + A;
  case R_GOT_PAGE_PC:
    return getAArch64Page(Body.getGotVA<ELFT>() + A) - getAArch64Page(P);
  case R_GOT_PC:
  case R_RELAX_TLS_GD_TO_IE_PC:
    return Body.getGotVA<ELFT>() + A - P;
  case R_GOTONLY_PC:
    return Out<ELFT>::Got->getVA() + A - P;
  case R_TLS:
    return Body.getVA<ELFT>(A) - Out<ELFT>::TlsPhdr->p_memsz;
  case R_NEG_TLS:
    return Out<ELF32LE>::TlsPhdr->p_memsz - Body.getVA<ELFT>(A);
  case R_ABS:
  case R_RELAX_TLS_GD_TO_LE:
  case R_RELAX_TLS_IE_TO_LE:
  case R_RELAX_TLS_LD_TO_LE:
    return Body.getVA<ELFT>(A);
  case R_GOT_OFF:
    return Body.getGotOffset<ELFT>() + A;
  case R_MIPS_GOT_LOCAL:
    // If relocation against MIPS local symbol requires GOT entry, this entry
    // should be initialized by 'page address'. This address is high 16-bits
    // of sum the symbol's value and the addend.
    return Out<ELFT>::Got->getMipsLocalPageOffset(Body.getVA<ELFT>(A));
  case R_MIPS_GOT:
    // For non-local symbols GOT entries should contain their full
    // addresses. But if such symbol cannot be preempted, we do not
    // have to put them into the "global" part of GOT and use dynamic
    // linker to determine their actual addresses. That is why we
    // create GOT entries for them in the "local" part of GOT.
    return Out<ELFT>::Got->getMipsLocalEntryOffset(Body.getVA<ELFT>(A));
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
    return Body.getVA<ELFT>(A) - P;
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
    uintX_t Addend = getAddend<ELFT>(Rel);
    if (!RelTy::IsRela)
      Addend += Target->getImplicitAddend(Buf + Rel.r_offset, Type);

    SymbolBody &Sym = this->File->getRelocTargetSym(Rel);
    if (Target->getRelExpr(Type, Sym) != R_ABS) {
      error(this->getSectionName() + " has non-ABS reloc");
      return;
    }

    uintX_t Offset = this->getOffset(Rel.r_offset);
    uint8_t *BufLoc = Buf + Offset;
    uintX_t AddrLoc = this->OutSec->getVA() + Offset;
    uint64_t SymVA = SignExtend64<Bits>(getSymVA<ELFT>(
        Type, Addend, AddrLoc, Sym, BufLoc, *this->File, R_ABS));
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
    uintX_t Offset = Rel.Offset;
    uint8_t *BufLoc = Buf + Offset;
    uint32_t Type = Rel.Type;
    uintX_t A = Rel.Addend;

    uintX_t AddrLoc = OutSec->getVA() + Offset;
    RelExpr Expr = Rel.Expr;
    uint64_t SymVA = SignExtend64<Bits>(
        getSymVA<ELFT>(Type, A, AddrLoc, *Rel.Sym, BufLoc, *File, Expr));

    if (Expr == R_RELAX_TLS_IE_TO_LE) {
      Target->relaxTlsIeToLe(BufLoc, Type, SymVA);
      continue;
    }
    if (Expr == R_RELAX_TLS_LD_TO_LE) {
      Target->relaxTlsLdToLe(BufLoc, Type, SymVA);
      continue;
    }
    if (Expr == R_RELAX_TLS_GD_TO_LE) {
      Target->relaxTlsGdToLe(BufLoc, Type, SymVA);
      continue;
    }
    if (Expr == R_RELAX_TLS_GD_TO_IE_PC || Expr == R_RELAX_TLS_GD_TO_IE) {
      Target->relaxTlsGdToIe(BufLoc, Type, SymVA);
      continue;
    }

    if (Expr == R_PPC_PLT_OPD) {
      uint32_t Nop = 0x60000000;
      if (BufLoc + 8 <= BufEnd && read32be(BufLoc + 4) == Nop)
        write32be(BufLoc + 4, 0xe8410028); // ld %r2, 40(%r1)
    }

    Target->relocateOne(BufLoc, Type, SymVA);
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
  ArrayRef<uint8_t> Data = this->getSectionData();
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
    for (const SymbolBody *S : Thunks) {
      Target->writeThunk(Buf, S->getVA<ELFT>());
      Buf += Target->ThunkSize;
    }
  }
}

template <class ELFT>
void InputSection<ELFT>::replace(InputSection<ELFT> *Other) {
  this->Align = std::max(this->Align, Other->Align);
  Other->Repl = this->Repl;
  Other->Live = false;
}

template <class ELFT>
SplitInputSection<ELFT>::SplitInputSection(
    elf::ObjectFile<ELFT> *File, const Elf_Shdr *Header,
    typename InputSectionBase<ELFT>::Kind SectionKind)
    : InputSectionBase<ELFT>(File, Header, SectionKind) {}

template <class ELFT>
EHInputSection<ELFT>::EHInputSection(elf::ObjectFile<ELFT> *F,
                                     const Elf_Shdr *Header)
    : SplitInputSection<ELFT>(F, Header, InputSectionBase<ELFT>::EHFrame) {
  // Mark .eh_frame sections as live by default because there are
  // usually no relocations that point to .eh_frames. Otherwise,
  // the garbage collector would drop all .eh_frame sections.
  this->Live = true;
}

template <class ELFT>
bool EHInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::EHFrame;
}

template <class ELFT>
typename ELFT::uint EHInputSection<ELFT>::getOffset(uintX_t Offset) {
  // The file crtbeginT.o has relocations pointing to the start of an empty
  // .eh_frame that is known to be the first in the link. It does that to
  // identify the start of the output .eh_frame. Handle this special case.
  if (this->getSectionHdr()->sh_size == 0)
    return Offset;
  std::pair<uintX_t, uintX_t> *I = this->getRangeAndSize(Offset).first;
  uintX_t Base = I->second;
  if (Base == uintX_t(-1))
    return -1; // Not in the output

  uintX_t Addend = Offset - I->first;
  return Base + Addend;
}

static size_t findNull(StringRef S, size_t EntSize) {
  // Optimize the common case.
  if (EntSize == 1)
    return S.find(0);

  for (unsigned I = 0, N = S.size(); I != N; I += EntSize) {
    const char *B = S.begin() + I;
    if (std::all_of(B, B + EntSize, [](char C) { return C == 0; }))
      return I;
  }
  return StringRef::npos;
}

template <class ELFT>
MergeInputSection<ELFT>::MergeInputSection(elf::ObjectFile<ELFT> *F,
                                           const Elf_Shdr *Header)
    : SplitInputSection<ELFT>(F, Header, InputSectionBase<ELFT>::Merge) {
  uintX_t EntSize = Header->sh_entsize;
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  std::vector<std::pair<uintX_t, uintX_t>> &Offsets = this->Offsets;

  uintX_t V = Config->GcSections ? -1 : 0;
  if (Header->sh_flags & SHF_STRINGS) {
    uintX_t Offset = 0;
    while (!Data.empty()) {
      size_t End = findNull(Data, EntSize);
      if (End == StringRef::npos)
        fatal("string is not null terminated");
      Offsets.push_back(std::make_pair(Offset, V));
      uintX_t Size = End + EntSize;
      Data = Data.substr(Size);
      Offset += Size;
    }
    return;
  }

  // If this is not of type string, every entry has the same size.
  size_t Size = Data.size();
  assert((Size % EntSize) == 0);
  for (unsigned I = 0, N = Size; I != N; I += EntSize)
    Offsets.push_back(std::make_pair(I, V));
}

template <class ELFT>
bool MergeInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::Merge;
}

template <class ELFT>
std::pair<std::pair<typename ELFT::uint, typename ELFT::uint> *,
          typename ELFT::uint>
SplitInputSection<ELFT>::getRangeAndSize(uintX_t Offset) {
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  uintX_t Size = Data.size();
  if (Offset >= Size)
    fatal("entry is past the end of the section");

  // Find the element this offset points to.
  auto I = std::upper_bound(
      Offsets.begin(), Offsets.end(), Offset,
      [](const uintX_t &A, const std::pair<uintX_t, uintX_t> &B) {
        return A < B.first;
      });
  uintX_t End = I == Offsets.end() ? Data.size() : I->first;
  --I;
  return std::make_pair(&*I, End);
}

template <class ELFT>
typename ELFT::uint MergeInputSection<ELFT>::getOffset(uintX_t Offset) {
  std::pair<std::pair<uintX_t, uintX_t> *, uintX_t> T =
      this->getRangeAndSize(Offset);
  std::pair<uintX_t, uintX_t> *I = T.first;
  uintX_t End = T.second;
  uintX_t Start = I->first;

  // Compute the Addend and if the Base is cached, return.
  uintX_t Addend = Offset - Start;
  uintX_t &Base = I->second;
  if (Base != uintX_t(-1))
    return Base + Addend;

  // Map the base to the offset in the output section and cache it.
  ArrayRef<uint8_t> D = this->getSectionData();
  StringRef Data((const char *)D.data(), D.size());
  StringRef Entry = Data.substr(Start, End - Start);
  Base =
      static_cast<MergeOutputSection<ELFT> *>(this->OutSec)->getOffset(Entry);
  return Base + Addend;
}

template <class ELFT>
MipsReginfoInputSection<ELFT>::MipsReginfoInputSection(elf::ObjectFile<ELFT> *F,
                                                       const Elf_Shdr *Hdr)
    : InputSectionBase<ELFT>(F, Hdr, InputSectionBase<ELFT>::MipsReginfo) {
  // Initialize this->Reginfo.
  ArrayRef<uint8_t> D = this->getSectionData();
  if (D.size() != sizeof(Elf_Mips_RegInfo<ELFT>))
    fatal("invalid size of .reginfo section");
  Reginfo = reinterpret_cast<const Elf_Mips_RegInfo<ELFT> *>(D.data());
}

template <class ELFT>
bool MipsReginfoInputSection<ELFT>::classof(const InputSectionBase<ELFT> *S) {
  return S->SectionKind == InputSectionBase<ELFT>::MipsReginfo;
}

template class elf::InputSectionBase<ELF32LE>;
template class elf::InputSectionBase<ELF32BE>;
template class elf::InputSectionBase<ELF64LE>;
template class elf::InputSectionBase<ELF64BE>;

template class elf::InputSection<ELF32LE>;
template class elf::InputSection<ELF32BE>;
template class elf::InputSection<ELF64LE>;
template class elf::InputSection<ELF64BE>;

template class elf::SplitInputSection<ELF32LE>;
template class elf::SplitInputSection<ELF32BE>;
template class elf::SplitInputSection<ELF64LE>;
template class elf::SplitInputSection<ELF64BE>;

template class elf::EHInputSection<ELF32LE>;
template class elf::EHInputSection<ELF32BE>;
template class elf::EHInputSection<ELF64LE>;
template class elf::EHInputSection<ELF64BE>;

template class elf::MergeInputSection<ELF32LE>;
template class elf::MergeInputSection<ELF32BE>;
template class elf::MergeInputSection<ELF64LE>;
template class elf::MergeInputSection<ELF64BE>;

template class elf::MipsReginfoInputSection<ELF32LE>;
template class elf::MipsReginfoInputSection<ELF32BE>;
template class elf::MipsReginfoInputSection<ELF64LE>;
template class elf::MipsReginfoInputSection<ELF64BE>;
