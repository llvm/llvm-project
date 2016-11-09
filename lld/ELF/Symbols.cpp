//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "Error.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "SyntheticSections.h"
#include "Target.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

template <class ELFT>
static typename ELFT::uint getSymVA(const SymbolBody &Body,
                                    typename ELFT::uint &Addend) {
  typedef typename ELFT::uint uintX_t;

  switch (Body.kind()) {
  case SymbolBody::DefinedSyntheticKind: {
    auto &D = cast<DefinedSynthetic<ELFT>>(Body);
    const OutputSectionBase<ELFT> *Sec = D.Section;
    if (!Sec)
      return D.Value;
    if (D.Value == DefinedSynthetic<ELFT>::SectionEnd)
      return Sec->Addr + Sec->Size;
    return Sec->Addr + D.Value;
  }
  case SymbolBody::DefinedRegularKind: {
    auto &D = cast<DefinedRegular<ELFT>>(Body);
    InputSectionBase<ELFT> *SC = D.Section;

    // According to the ELF spec reference to a local symbol from outside
    // the group are not allowed. Unfortunately .eh_frame breaks that rule
    // and must be treated specially. For now we just replace the symbol with
    // 0.
    if (SC == &InputSection<ELFT>::Discarded)
      return 0;

    // This is an absolute symbol.
    if (!SC)
      return D.Value;

    uintX_t Offset = D.Value;
    if (D.isSection()) {
      Offset += Addend;
      Addend = 0;
    }
    uintX_t VA = (SC->OutSec ? SC->OutSec->Addr : 0) + SC->getOffset(Offset);
    if (D.isTls() && !Config->Relocatable) {
      if (!Out<ELFT>::TlsPhdr)
        fatal(getFilename(D.File) +
              " has a STT_TLS symbol but doesn't have a PT_TLS section");
      return VA - Out<ELFT>::TlsPhdr->p_vaddr;
    }
    return VA;
  }
  case SymbolBody::DefinedCommonKind:
    return In<ELFT>::Common->OutSec->Addr + In<ELFT>::Common->OutSecOff +
           cast<DefinedCommon>(Body).Offset;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(Body);
    if (!SS.NeedsCopyOrPltAddr)
      return 0;
    if (SS.isFunc())
      return Body.getPltVA<ELFT>();
    return Out<ELFT>::Bss->Addr + SS.OffsetInBss;
  }
  case SymbolBody::UndefinedKind:
    return 0;
  case SymbolBody::LazyArchiveKind:
  case SymbolBody::LazyObjectKind:
    assert(Body.symbol()->IsUsedInRegularObj && "lazy symbol reached writer");
    return 0;
  }
  llvm_unreachable("invalid symbol kind");
}

SymbolBody::SymbolBody(Kind K, uint32_t NameOffset, uint8_t StOther,
                       uint8_t Type)
    : SymbolKind(K), NeedsCopyOrPltAddr(false), IsLocal(true),
      IsInGlobalMipsGot(false), Is32BitMipsGot(false), Type(Type),
      StOther(StOther), NameOffset(NameOffset) {}

SymbolBody::SymbolBody(Kind K, StringRef Name, uint8_t StOther, uint8_t Type)
    : SymbolKind(K), NeedsCopyOrPltAddr(false), IsLocal(false),
      IsInGlobalMipsGot(false), Is32BitMipsGot(false), Type(Type),
      StOther(StOther), Name({Name.data(), Name.size()}) {}

StringRef SymbolBody::getName() const {
  assert(!isLocal());
  return StringRef(Name.S, Name.Len);
}

// Returns true if a symbol can be replaced at load-time by a symbol
// with the same name defined in other ELF executable or DSO.
bool SymbolBody::isPreemptible() const {
  if (isLocal())
    return false;

  // Shared symbols resolve to the definition in the DSO. The exceptions are
  // symbols with copy relocations (which resolve to .bss) or preempt plt
  // entries (which resolve to that plt entry).
  if (isShared())
    return !NeedsCopyOrPltAddr;

  // That's all that can be preempted in a non-DSO.
  if (!Config->Shared)
    return false;

  // Only symbols that appear in dynsym can be preempted.
  if (!symbol()->includeInDynsym())
    return false;

  // Only default visibility symbols can be preempted.
  if (symbol()->Visibility != STV_DEFAULT)
    return false;

  // -Bsymbolic means that definitions are not preempted.
  if (Config->Bsymbolic || (Config->BsymbolicFunctions && isFunc()))
    return !isDefined();
  return true;
}

template <class ELFT> bool SymbolBody::hasThunk() const {
  if (auto *DR = dyn_cast<DefinedRegular<ELFT>>(this))
    return DR->ThunkData != nullptr;
  if (auto *S = dyn_cast<SharedSymbol<ELFT>>(this))
    return S->ThunkData != nullptr;
  return false;
}

template <class ELFT>
typename ELFT::uint SymbolBody::getVA(typename ELFT::uint Addend) const {
  typename ELFT::uint OutVA = getSymVA<ELFT>(*this, Addend);
  return OutVA + Addend;
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotVA() const {
  return Out<ELFT>::Got->Addr + getGotOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotOffset() const {
  return GotIndex * Target->GotEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltVA() const {
  return Out<ELFT>::GotPlt->Addr + getGotPltOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltOffset() const {
  return GotPltIndex * Target->GotPltEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getPltVA() const {
  return Out<ELFT>::Plt->Addr + Target->PltHeaderSize +
         PltIndex * Target->PltEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getThunkVA() const {
  if (const auto *DR = dyn_cast<DefinedRegular<ELFT>>(this))
    return DR->ThunkData->getVA();
  if (const auto *S = dyn_cast<SharedSymbol<ELFT>>(this))
    return S->ThunkData->getVA();
  fatal("getThunkVA() not supported for Symbol class\n");
}

template <class ELFT> typename ELFT::uint SymbolBody::getSize() const {
  if (const auto *C = dyn_cast<DefinedCommon>(this))
    return C->Size;
  if (const auto *DR = dyn_cast<DefinedRegular<ELFT>>(this))
    return DR->Size;
  if (const auto *S = dyn_cast<SharedSymbol<ELFT>>(this))
    return S->Sym.st_size;
  return 0;
}

Defined::Defined(Kind K, StringRef Name, uint8_t StOther, uint8_t Type)
    : SymbolBody(K, Name, StOther, Type) {}

Defined::Defined(Kind K, uint32_t NameOffset, uint8_t StOther, uint8_t Type)
    : SymbolBody(K, NameOffset, StOther, Type) {}

template <class ELFT> bool DefinedRegular<ELFT>::isMipsPIC() const {
  if (!Section || !isFunc())
    return false;
  return (this->StOther & STO_MIPS_MIPS16) == STO_MIPS_PIC ||
         (Section->getFile()->getObj().getHeader()->e_flags & EF_MIPS_PIC);
}

Undefined::Undefined(StringRef Name, uint8_t StOther, uint8_t Type,
                     InputFile *File)
    : SymbolBody(SymbolBody::UndefinedKind, Name, StOther, Type) {
  this->File = File;
}

Undefined::Undefined(uint32_t NameOffset, uint8_t StOther, uint8_t Type,
                     InputFile *File)
    : SymbolBody(SymbolBody::UndefinedKind, NameOffset, StOther, Type) {
  this->File = File;
}

template <typename ELFT>
DefinedSynthetic<ELFT>::DefinedSynthetic(StringRef N, uintX_t Value,
                                         OutputSectionBase<ELFT> *Section)
    : Defined(SymbolBody::DefinedSyntheticKind, N, STV_HIDDEN, 0 /* Type */),
      Value(Value), Section(Section) {}

DefinedCommon::DefinedCommon(StringRef N, uint64_t Size, uint64_t Alignment,
                             uint8_t StOther, uint8_t Type, InputFile *File)
    : Defined(SymbolBody::DefinedCommonKind, N, StOther, Type),
      Alignment(Alignment), Size(Size) {
  this->File = File;
}

InputFile *Lazy::fetch() {
  if (auto *S = dyn_cast<LazyArchive>(this))
    return S->fetch();
  return cast<LazyObject>(this)->fetch();
}

LazyArchive::LazyArchive(ArchiveFile &File,
                         const llvm::object::Archive::Symbol S, uint8_t Type)
    : Lazy(LazyArchiveKind, S.getName(), Type), Sym(S) {
  this->File = &File;
}

LazyObject::LazyObject(StringRef Name, LazyObjectFile &File, uint8_t Type)
    : Lazy(LazyObjectKind, Name, Type) {
  this->File = &File;
}

InputFile *LazyArchive::fetch() {
  std::pair<MemoryBufferRef, uint64_t> MBInfo = file()->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBInfo.first.getBuffer().empty())
    return nullptr;
  return createObjectFile(MBInfo.first, file()->getName(), MBInfo.second);
}

InputFile *LazyObject::fetch() {
  MemoryBufferRef MBRef = file()->getBuffer();
  if (MBRef.getBuffer().empty())
    return nullptr;
  return createObjectFile(MBRef);
}

bool Symbol::includeInDynsym() const {
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return false;
  return (ExportDynamic && VersionId != VER_NDX_LOCAL) || body()->isShared() ||
         (body()->isUndefined() && Config->Shared);
}

// Print out a log message for --trace-symbol.
void elf::printTraceSymbol(Symbol *Sym) {
  SymbolBody *B = Sym->body();
  outs() << getFilename(B->File);

  if (B->isUndefined())
    outs() << ": reference to ";
  else if (B->isCommon())
    outs() << ": common definition of ";
  else
    outs() << ": definition of ";
  outs() << B->getName() << "\n";
}

template bool SymbolBody::hasThunk<ELF32LE>() const;
template bool SymbolBody::hasThunk<ELF32BE>() const;
template bool SymbolBody::hasThunk<ELF64LE>() const;
template bool SymbolBody::hasThunk<ELF64BE>() const;

template uint32_t SymbolBody::template getVA<ELF32LE>(uint32_t) const;
template uint32_t SymbolBody::template getVA<ELF32BE>(uint32_t) const;
template uint64_t SymbolBody::template getVA<ELF64LE>(uint64_t) const;
template uint64_t SymbolBody::template getVA<ELF64BE>(uint64_t) const;

template uint32_t SymbolBody::template getGotVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotOffset<ELF32LE>() const;
template uint32_t SymbolBody::template getGotOffset<ELF32BE>() const;
template uint64_t SymbolBody::template getGotOffset<ELF64LE>() const;
template uint64_t SymbolBody::template getGotOffset<ELF64BE>() const;

template uint32_t SymbolBody::template getGotPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getGotPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getGotPltVA<ELF64BE>() const;

template uint32_t SymbolBody::template getThunkVA<ELF32LE>() const;
template uint32_t SymbolBody::template getThunkVA<ELF32BE>() const;
template uint64_t SymbolBody::template getThunkVA<ELF64LE>() const;
template uint64_t SymbolBody::template getThunkVA<ELF64BE>() const;

template uint32_t SymbolBody::template getGotPltOffset<ELF32LE>() const;
template uint32_t SymbolBody::template getGotPltOffset<ELF32BE>() const;
template uint64_t SymbolBody::template getGotPltOffset<ELF64LE>() const;
template uint64_t SymbolBody::template getGotPltOffset<ELF64BE>() const;

template uint32_t SymbolBody::template getPltVA<ELF32LE>() const;
template uint32_t SymbolBody::template getPltVA<ELF32BE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64LE>() const;
template uint64_t SymbolBody::template getPltVA<ELF64BE>() const;

template uint32_t SymbolBody::template getSize<ELF32LE>() const;
template uint32_t SymbolBody::template getSize<ELF32BE>() const;
template uint64_t SymbolBody::template getSize<ELF64LE>() const;
template uint64_t SymbolBody::template getSize<ELF64BE>() const;

template class elf::DefinedRegular<ELF32LE>;
template class elf::DefinedRegular<ELF32BE>;
template class elf::DefinedRegular<ELF64LE>;
template class elf::DefinedRegular<ELF64BE>;

template class elf::DefinedSynthetic<ELF32LE>;
template class elf::DefinedSynthetic<ELF32BE>;
template class elf::DefinedSynthetic<ELF64LE>;
template class elf::DefinedSynthetic<ELF64BE>;
