//===- Symbols.cpp --------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "Strings.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <cstring>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

DefinedRegular *ElfSym::Bss;
DefinedRegular *ElfSym::Etext1;
DefinedRegular *ElfSym::Etext2;
DefinedRegular *ElfSym::Edata1;
DefinedRegular *ElfSym::Edata2;
DefinedRegular *ElfSym::End1;
DefinedRegular *ElfSym::End2;
DefinedRegular *ElfSym::GlobalOffsetTable;
DefinedRegular *ElfSym::MipsGp;
DefinedRegular *ElfSym::MipsGpDisp;
DefinedRegular *ElfSym::MipsLocalGp;

static uint64_t getSymVA(const Symbol &Sym, int64_t &Addend) {
  switch (Sym.kind()) {
  case Symbol::DefinedRegularKind: {
    auto &D = cast<DefinedRegular>(Sym);
    SectionBase *IS = D.Section;
    if (auto *ISB = dyn_cast_or_null<InputSectionBase>(IS))
      IS = ISB->Repl;

    // According to the ELF spec reference to a local symbol from outside
    // the group are not allowed. Unfortunately .eh_frame breaks that rule
    // and must be treated specially. For now we just replace the symbol with
    // 0.
    if (IS == &InputSection::Discarded)
      return 0;

    // This is an absolute symbol.
    if (!IS)
      return D.Value;

    uint64_t Offset = D.Value;

    // An object in an SHF_MERGE section might be referenced via a
    // section symbol (as a hack for reducing the number of local
    // symbols).
    // Depending on the addend, the reference via a section symbol
    // refers to a different object in the merge section.
    // Since the objects in the merge section are not necessarily
    // contiguous in the output, the addend can thus affect the final
    // VA in a non-linear way.
    // To make this work, we incorporate the addend into the section
    // offset (and zero out the addend for later processing) so that
    // we find the right object in the section.
    if (D.isSection()) {
      Offset += Addend;
      Addend = 0;
    }

    const OutputSection *OutSec = IS->getOutputSection();

    // In the typical case, this is actually very simple and boils
    // down to adding together 3 numbers:
    // 1. The address of the output section.
    // 2. The offset of the input section within the output section.
    // 3. The offset within the input section (this addition happens
    //    inside InputSection::getOffset).
    //
    // If you understand the data structures involved with this next
    // line (and how they get built), then you have a pretty good
    // understanding of the linker.
    uint64_t VA = (OutSec ? OutSec->Addr : 0) + IS->getOffset(Offset);

    if (D.isTls() && !Config->Relocatable) {
      if (!Out::TlsPhdr)
        fatal(toString(D.getFile()) +
              " has an STT_TLS symbol but doesn't have an SHF_TLS section");
      return VA - Out::TlsPhdr->p_vaddr;
    }
    return VA;
  }
  case Symbol::DefinedCommonKind:
    llvm_unreachable("common are converted to bss");
  case Symbol::SharedKind: {
    auto &SS = cast<SharedSymbol>(Sym);
    if (SS.CopyRelSec)
      return SS.CopyRelSec->getParent()->Addr + SS.CopyRelSec->OutSecOff;
    if (SS.NeedsPltAddr)
      return Sym.getPltVA();
    return 0;
  }
  case Symbol::UndefinedKind:
    return 0;
  case Symbol::LazyArchiveKind:
  case Symbol::LazyObjectKind:
    assert(Sym.IsUsedInRegularObj && "lazy symbol reached writer");
    return 0;
  }
  llvm_unreachable("invalid symbol kind");
}

// Returns true if this is a weak undefined symbol.
bool Symbol::isUndefWeak() const {
  // See comment on Lazy in Symbols.h for the details.
  return !isLocal() && isWeak() && (isUndefined() || isLazy());
}

InputFile *Symbol::getFile() const {
  if (isLocal()) {
    const SectionBase *Sec = cast<DefinedRegular>(this)->Section;
    // Local absolute symbols actually have a file, but that is not currently
    // used. We could support that by having a mostly redundant InputFile in
    // Symbol, or having a special absolute section if needed.
    return Sec ? cast<InputSectionBase>(Sec)->File : nullptr;
  }
  return File;
}

// Overwrites all attributes with Other's so that this symbol becomes
// an alias to Other. This is useful for handling some options such as
// --wrap.
void Symbol::copyFrom(Symbol *Other) {
  Symbol Sym = *this;
  memcpy(this, Other, sizeof(SymbolUnion));

  Binding = Sym.Binding;
  VersionId = Sym.VersionId;
  Visibility = Sym.Visibility;
  IsUsedInRegularObj = Sym.IsUsedInRegularObj;
  ExportDynamic = Sym.ExportDynamic;
  CanInline = Sym.CanInline;
  Traced = Sym.Traced;
  InVersionScript = Sym.InVersionScript;
}

uint64_t Symbol::getVA(int64_t Addend) const {
  uint64_t OutVA = getSymVA(*this, Addend);
  return OutVA + Addend;
}

uint64_t Symbol::getGotVA() const { return InX::Got->getVA() + getGotOffset(); }

uint64_t Symbol::getGotOffset() const {
  return GotIndex * Target->GotEntrySize;
}

uint64_t Symbol::getGotPltVA() const {
  if (this->IsInIgot)
    return InX::IgotPlt->getVA() + getGotPltOffset();
  return InX::GotPlt->getVA() + getGotPltOffset();
}

uint64_t Symbol::getGotPltOffset() const {
  return GotPltIndex * Target->GotPltEntrySize;
}

uint64_t Symbol::getPltVA() const {
  if (this->IsInIplt)
    return InX::Iplt->getVA() + PltIndex * Target->PltEntrySize;
  return InX::Plt->getVA() + Target->PltHeaderSize +
         PltIndex * Target->PltEntrySize;
}

uint64_t Symbol::getSize() const {
  if (const auto *C = dyn_cast<DefinedCommon>(this))
    return C->Size;
  if (const auto *DR = dyn_cast<DefinedRegular>(this))
    return DR->Size;
  if (const auto *S = dyn_cast<SharedSymbol>(this))
    return S->Size;
  return 0;
}

OutputSection *Symbol::getOutputSection() const {
  if (auto *S = dyn_cast<DefinedRegular>(this)) {
    if (S->Section)
      return S->Section->getOutputSection();
    return nullptr;
  }

  if (auto *S = dyn_cast<SharedSymbol>(this)) {
    if (S->CopyRelSec)
      return S->CopyRelSec->getParent();
    return nullptr;
  }

  if (auto *S = dyn_cast<DefinedCommon>(this)) {
    if (Config->DefineCommon)
      return S->Section->getParent();
    return nullptr;
  }

  return nullptr;
}

// If a symbol name contains '@', the characters after that is
// a symbol version name. This function parses that.
void Symbol::parseSymbolVersion() {
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
  if (!isInCurrentOutput())
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
      VersionId = Ver.Id;
    else
      VersionId = Ver.Id | VERSYM_HIDDEN;
    return;
  }

  // It is an error if the specified version is not defined.
  // Usually version script is not provided when linking executable,
  // but we may still want to override a versioned symbol from DSO,
  // so we do not report error in this case.
  if (Config->Shared)
    error(toString(getFile()) + ": symbol " + S + " has undefined version " +
          Verstr);
}

template <class ELFT> bool DefinedRegular::isMipsPIC() const {
  typedef typename ELFT::Ehdr Elf_Ehdr;
  if (!Section || !isFunc())
    return false;

  auto *Sec = cast<InputSectionBase>(Section);
  const Elf_Ehdr *Hdr = Sec->template getFile<ELFT>()->getObj().getHeader();
  return (this->StOther & STO_MIPS_MIPS16) == STO_MIPS_PIC ||
         (Hdr->e_flags & EF_MIPS_PIC);
}

InputFile *Lazy::fetch() {
  if (auto *S = dyn_cast<LazyArchive>(this))
    return S->fetch();
  return cast<LazyObject>(this)->fetch();
}

ArchiveFile *LazyArchive::getFile() {
  return cast<ArchiveFile>(Symbol::getFile());
}

InputFile *LazyArchive::fetch() {
  std::pair<MemoryBufferRef, uint64_t> MBInfo = getFile()->getMember(&Sym);

  // getMember returns an empty buffer if the member was already
  // read from the library.
  if (MBInfo.first.getBuffer().empty())
    return nullptr;
  return createObjectFile(MBInfo.first, getFile()->getName(), MBInfo.second);
}

LazyObjFile *LazyObject::getFile() {
  return cast<LazyObjFile>(Symbol::getFile());
}

InputFile *LazyObject::fetch() { return getFile()->fetch(); }

uint8_t Symbol::computeBinding() const {
  if (Config->Relocatable)
    return Binding;
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return STB_LOCAL;
  if (VersionId == VER_NDX_LOCAL && isInCurrentOutput())
    return STB_LOCAL;
  if (Config->NoGnuUnique && Binding == STB_GNU_UNIQUE)
    return STB_GLOBAL;
  return Binding;
}

bool Symbol::includeInDynsym() const {
  if (!Config->HasDynSymTab)
    return false;
  if (computeBinding() == STB_LOCAL)
    return false;
  if (!isInCurrentOutput())
    return true;
  return ExportDynamic;
}

// Print out a log message for --trace-symbol.
void elf::printTraceSymbol(Symbol *Sym) {
  std::string S;
  if (Sym->isUndefined())
    S = ": reference to ";
  else if (Sym->isCommon())
    S = ": common definition of ";
  else if (Sym->isLazy())
    S = ": lazy definition of ";
  else if (Sym->isShared())
    S = ": shared definition of ";
  else
    S = ": definition of ";

  message(toString(Sym->File) + S + Sym->getName());
}

// Returns a symbol for an error message.
std::string lld::toString(const Symbol &B) {
  if (Config->Demangle)
    if (Optional<std::string> S = demangle(B.getName()))
      return *S;
  return B.getName();
}

template bool DefinedRegular::template isMipsPIC<ELF32LE>() const;
template bool DefinedRegular::template isMipsPIC<ELF32BE>() const;
template bool DefinedRegular::template isMipsPIC<ELF64LE>() const;
template bool DefinedRegular::template isMipsPIC<ELF64BE>() const;
