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
#include "Strings.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <cstring>

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
    auto &D = cast<DefinedSynthetic>(Body);
    const OutputSectionBase *Sec = D.Section;
    if (!Sec)
      return D.Value;
    if (D.Value == uintX_t(-1))
      return Sec->Addr + Sec->Size;
    return Sec->Addr + D.Value;
  }
  case SymbolBody::DefinedRegularKind: {
    auto &D = cast<DefinedRegular<ELFT>>(Body);
    InputSectionBase<ELFT> *IS = D.Section;

    // According to the ELF spec reference to a local symbol from outside
    // the group are not allowed. Unfortunately .eh_frame breaks that rule
    // and must be treated specially. For now we just replace the symbol with
    // 0.
    if (IS == &InputSection<ELFT>::Discarded)
      return 0;

    // This is an absolute symbol.
    if (!IS)
      return D.Value;

    uintX_t Offset = D.Value;
    if (D.isSection()) {
      Offset += Addend;
      Addend = 0;
    }
    const OutputSectionBase *OutSec = IS->getOutputSection();
    uintX_t VA = (OutSec ? OutSec->Addr : 0) + IS->getOffset(Offset);
    if (D.isTls() && !Config->Relocatable) {
      if (!Out<ELFT>::TlsPhdr)
        fatal(toString(D.File) +
              " has a STT_TLS symbol but doesn't have a PT_TLS section");
      return VA - Out<ELFT>::TlsPhdr->p_vaddr;
    }
    return VA;
  }
  case SymbolBody::DefinedCommonKind:
    if (!Config->DefineCommon)
      return 0;
    return In<ELFT>::Common->OutSec->Addr + In<ELFT>::Common->OutSecOff +
           cast<DefinedCommon>(Body).Offset;
  case SymbolBody::SharedKind: {
    auto &SS = cast<SharedSymbol<ELFT>>(Body);
    if (!SS.NeedsCopyOrPltAddr)
      return 0;
    if (SS.isFunc())
      return Body.getPltVA<ELFT>();
    InputSection<ELFT> *CopyISec = SS.getBssSectionForCopy();
    return CopyISec->OutSec->Addr + CopyISec->OutSecOff;
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

SymbolBody::SymbolBody(Kind K, StringRefZ Name, bool IsLocal, uint8_t StOther,
                       uint8_t Type)
    : SymbolKind(K), NeedsCopyOrPltAddr(false), IsLocal(IsLocal),
      IsInGlobalMipsGot(false), Is32BitMipsGot(false), IsInIplt(false),
      IsInIgot(false), Type(Type), StOther(StOther),
      Name(Name) {}

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

template <class ELFT>
typename ELFT::uint SymbolBody::getVA(typename ELFT::uint Addend) const {
  typename ELFT::uint OutVA = getSymVA<ELFT>(*this, Addend);
  return OutVA + Addend;
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotVA() const {
  return In<ELFT>::Got->getVA() + getGotOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotOffset() const {
  return GotIndex * Target->GotEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltVA() const {
  if (this->IsInIgot)
    return In<ELFT>::IgotPlt->getVA() + getGotPltOffset<ELFT>();
  return In<ELFT>::GotPlt->getVA() + getGotPltOffset<ELFT>();
}

template <class ELFT> typename ELFT::uint SymbolBody::getGotPltOffset() const {
  return GotPltIndex * Target->GotPltEntrySize;
}

template <class ELFT> typename ELFT::uint SymbolBody::getPltVA() const {
  if (this->IsInIplt)
    return In<ELFT>::Iplt->getVA() + PltIndex * Target->PltEntrySize;
  return In<ELFT>::Plt->getVA() + Target->PltHeaderSize +
         PltIndex * Target->PltEntrySize;
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

// If a symbol name contains '@', the characters after that is
// a symbol version name. This function parses that.
void SymbolBody::parseSymbolVersion() {
  StringRef S = getName();
  size_t Pos = S.find('@');
  if (Pos == 0 || Pos == StringRef::npos)
    return;
  StringRef Verstr = S.substr(Pos + 1);
  if (Verstr.empty())
    return;

  // Truncate the symbol name so that it doesn't include the version string.
  Name = {S.data(), Pos};

  // If this is not in this DSO, it is not a definition.
  if (!isInCurrentDSO())
    return;

  // '@@' in a symbol name means the default version.
  // It is usually the most recent one.
  bool IsDefault = (Verstr[0] == '@');
  if (IsDefault)
    Verstr = Verstr.substr(1);

  for (VersionDefinition &Ver : Config->VersionDefinitions) {
    if (Ver.Name != Verstr)
      continue;

    if (IsDefault)
      symbol()->VersionId = Ver.Id;
    else
      symbol()->VersionId = Ver.Id | VERSYM_HIDDEN;
    return;
  }

  // It is an error if the specified version is not defined.
  error(toString(File) + ": symbol " + S + " has undefined version " + Verstr);
}

Defined::Defined(Kind K, StringRefZ Name, bool IsLocal, uint8_t StOther,
                 uint8_t Type)
    : SymbolBody(K, Name, IsLocal, StOther, Type) {}

template <class ELFT> bool DefinedRegular<ELFT>::isMipsPIC() const {
  if (!Section || !isFunc())
    return false;
  return (this->StOther & STO_MIPS_MIPS16) == STO_MIPS_PIC ||
         (Section->getFile()->getObj().getHeader()->e_flags & EF_MIPS_PIC);
}

Undefined::Undefined(StringRefZ Name, bool IsLocal, uint8_t StOther,
                     uint8_t Type, InputFile *File)
    : SymbolBody(SymbolBody::UndefinedKind, Name, IsLocal, StOther, Type) {
  this->File = File;
}

template <typename ELFT>
InputSection<ELFT> *SharedSymbol<ELFT>::getBssSectionForCopy() const {
  assert(needsCopy());
  assert(CopySection);
  return CopySection;
}

DefinedCommon::DefinedCommon(StringRef Name, uint64_t Size, uint64_t Alignment,
                             uint8_t StOther, uint8_t Type, InputFile *File)
    : Defined(SymbolBody::DefinedCommonKind, Name, /*IsLocal=*/false, StOther,
              Type),
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

uint8_t Symbol::computeBinding() const {
  if (Config->Relocatable)
    return Binding;
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return STB_LOCAL;
  const SymbolBody *Body = body();
  if (VersionId == VER_NDX_LOCAL && Body->isInCurrentDSO())
    return STB_LOCAL;
  if (Config->NoGnuUnique && Binding == STB_GNU_UNIQUE)
    return STB_GLOBAL;
  return Binding;
}

bool Symbol::includeInDynsym() const {
  if (computeBinding() == STB_LOCAL)
    return false;
  return ExportDynamic || body()->isShared() ||
         (body()->isUndefined() && Config->Shared);
}

// Print out a log message for --trace-symbol.
void elf::printTraceSymbol(Symbol *Sym) {
  SymbolBody *B = Sym->body();
  outs() << toString(B->File);

  if (B->isUndefined())
    outs() << ": reference to ";
  else if (B->isCommon())
    outs() << ": common definition of ";
  else
    outs() << ": definition of ";
  outs() << B->getName() << "\n";
}

// Returns a symbol for an error message.
std::string lld::toString(const SymbolBody &B) {
  if (Config->Demangle)
    if (Optional<std::string> S = demangle(B.getName()))
      return *S;
  return B.getName();
}

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

template class elf::SharedSymbol<ELF32LE>;
template class elf::SharedSymbol<ELF32BE>;
template class elf::SharedSymbol<ELF64LE>;
template class elf::SharedSymbol<ELF64BE>;

template class elf::DefinedRegular<ELF32LE>;
template class elf::DefinedRegular<ELF32BE>;
template class elf::DefinedRegular<ELF64LE>;
template class elf::DefinedRegular<ELF64BE>;
