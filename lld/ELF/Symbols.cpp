//===- Symbols.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Symbols.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "OutputSections.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "Writer.h"
#include "lld/Common/ErrorHandler.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Path.h"
#include <cstring>

using namespace llvm;
using namespace llvm::object;
using namespace llvm::ELF;

using namespace lld;
using namespace lld::elf;

Defined *ElfSym::Bss;
Defined *ElfSym::Etext1;
Defined *ElfSym::Etext2;
Defined *ElfSym::Edata1;
Defined *ElfSym::Edata2;
Defined *ElfSym::End1;
Defined *ElfSym::End2;
Defined *ElfSym::GlobalOffsetTable;
Defined *ElfSym::MipsGp;
Defined *ElfSym::MipsGpDisp;
Defined *ElfSym::MipsLocalGp;
Defined *ElfSym::RelaIpltStart;
Defined *ElfSym::RelaIpltEnd;
Defined *ElfSym::RISCVGlobalPointer;
Defined *ElfSym::TlsModuleBase;

static uint64_t getSymVA(const Symbol &Sym, int64_t &Addend) {
  switch (Sym.kind()) {
  case Symbol::DefinedKind: {
    auto &D = cast<Defined>(Sym);
    SectionBase *IS = D.Section;

    // This is an absolute symbol.
    if (!IS)
      return D.Value;

    assert(IS != &InputSection::Discarded);
    IS = IS->Repl;

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
    uint64_t VA = IS->getVA(Offset);

    // MIPS relocatable files can mix regular and microMIPS code.
    // Linker needs to distinguish such code. To do so microMIPS
    // symbols has the `STO_MIPS_MICROMIPS` flag in the `st_other`
    // field. Unfortunately, the `MIPS::relocateOne()` method has
    // a symbol value only. To pass type of the symbol (regular/microMIPS)
    // to that routine as well as other places where we write
    // a symbol value as-is (.dynamic section, `Elf_Ehdr::e_entry`
    // field etc) do the same trick as compiler uses to mark microMIPS
    // for CPU - set the less-significant bit.
    if (Config->EMachine == EM_MIPS && isMicroMips() &&
        ((Sym.StOther & STO_MIPS_MICROMIPS) || Sym.NeedsPltAddr))
      VA |= 1;

    if (D.isTls() && !Config->Relocatable) {
      // Use the address of the TLS segment's first section rather than the
      // segment's address, because segment addresses aren't initialized until
      // after sections are finalized. (e.g. Measuring the size of .rela.dyn
      // for Android relocation packing requires knowing TLS symbol addresses
      // during section finalization.)
      if (!Out::TlsPhdr || !Out::TlsPhdr->FirstSec)
        fatal(toString(D.File) +
              " has an STT_TLS symbol but doesn't have an SHF_TLS section");
      return VA - Out::TlsPhdr->FirstSec->Addr;
    }
    return VA;
  }
  case Symbol::SharedKind:
  case Symbol::UndefinedKind:
    return 0;
  case Symbol::LazyArchiveKind:
  case Symbol::LazyObjectKind:
    assert(Sym.IsUsedInRegularObj && "lazy symbol reached writer");
    return 0;
  case Symbol::CommonKind:
    llvm_unreachable("common symbol reached writer");
  case Symbol::PlaceholderKind:
    llvm_unreachable("placeholder symbol reached writer");
  }
  llvm_unreachable("invalid symbol kind");
}

uint64_t Symbol::getVA(int64_t Addend) const {
  uint64_t OutVA = getSymVA(*this, Addend);
  return OutVA + Addend;
}

uint64_t Symbol::getGotVA() const {
  if (GotInIgot)
    return In.IgotPlt->getVA() + getGotPltOffset();
  return In.Got->getVA() + getGotOffset();
}

uint64_t Symbol::getGotOffset() const { return GotIndex * Config->Wordsize; }

uint64_t Symbol::getGotPltVA() const {
  if (IsInIplt)
    return In.IgotPlt->getVA() + getGotPltOffset();
  return In.GotPlt->getVA() + getGotPltOffset();
}

uint64_t Symbol::getGotPltOffset() const {
  if (IsInIplt)
    return PltIndex * Config->Wordsize;
  return (PltIndex + Target->GotPltHeaderEntriesNum) * Config->Wordsize;
}

uint64_t Symbol::getPPC64LongBranchOffset() const {
  assert(PPC64BranchltIndex != 0xffff);
  return PPC64BranchltIndex * Config->Wordsize;
}

uint64_t Symbol::getPltVA() const {
  PltSection *Plt = IsInIplt ? In.Iplt : In.Plt;
  uint64_t OutVA =
      Plt->getVA() + Plt->HeaderSize + PltIndex * Target->PltEntrySize;
  // While linking microMIPS code PLT code are always microMIPS
  // code. Set the less-significant bit to track that fact.
  // See detailed comment in the `getSymVA` function.
  if (Config->EMachine == EM_MIPS && isMicroMips())
    OutVA |= 1;
  return OutVA;
}

uint64_t Symbol::getPPC64LongBranchTableVA() const {
  assert(PPC64BranchltIndex != 0xffff);
  return In.PPC64LongBranchTarget->getVA() +
         PPC64BranchltIndex * Config->Wordsize;
}

uint64_t Symbol::getSize() const {
  if (const auto *DR = dyn_cast<Defined>(this))
    return DR->Size;
  return cast<SharedSymbol>(this)->Size;
}

OutputSection *Symbol::getOutputSection() const {
  if (auto *S = dyn_cast<Defined>(this)) {
    if (auto *Sec = S->Section)
      return Sec->Repl->getOutputSection();
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
  NameSize = Pos;

  // If this is not in this DSO, it is not a definition.
  if (!isDefined())
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
  // so we do not report error in this case. We also do not error
  // if the symbol has a local version as it won't be in the dynamic
  // symbol table.
  if (Config->Shared && VersionId != VER_NDX_LOCAL)
    error(toString(File) + ": symbol " + S + " has undefined version " +
          Verstr);
}

void Symbol::fetch() const {
  if (auto *Sym = dyn_cast<LazyArchive>(this)) {
    cast<ArchiveFile>(Sym->File)->fetch(Sym->Sym);
    return;
  }

  if (auto *Sym = dyn_cast<LazyObject>(this)) {
    dyn_cast<LazyObjFile>(Sym->File)->fetch();
    return;
  }

  llvm_unreachable("Symbol::fetch() is called on a non-lazy symbol");
}

MemoryBufferRef LazyArchive::getMemberBuffer() {
  Archive::Child C = CHECK(
      Sym.getMember(), "could not get the member for symbol " + Sym.getName());

  return CHECK(C.getMemoryBufferRef(),
               "could not get the buffer for the member defining symbol " +
                   Sym.getName());
}

uint8_t Symbol::computeBinding() const {
  if (Config->Relocatable)
    return Binding;
  if (Visibility != STV_DEFAULT && Visibility != STV_PROTECTED)
    return STB_LOCAL;
  if (VersionId == VER_NDX_LOCAL && isDefined() && !IsPreemptible)
    return STB_LOCAL;
  if (!Config->GnuUnique && Binding == STB_GNU_UNIQUE)
    return STB_GLOBAL;
  return Binding;
}

bool Symbol::includeInDynsym() const {
  if (!Config->HasDynSymTab)
    return false;
  if (computeBinding() == STB_LOCAL)
    return false;

  // If a PIE binary was not linked against any shared libraries, then we can
  // safely drop weak undef symbols from .dynsym.
  if (isUndefWeak() && Config->Pie && SharedFiles.empty())
    return false;

  return isUndefined() || isShared() || ExportDynamic;
}

// Print out a log message for --trace-symbol.
void elf::printTraceSymbol(const Symbol *Sym) {
  std::string S;
  if (Sym->isUndefined())
    S = ": reference to ";
  else if (Sym->isLazy())
    S = ": lazy definition of ";
  else if (Sym->isShared())
    S = ": shared definition of ";
  else if (Sym->isCommon())
    S = ": common definition of ";
  else
    S = ": definition of ";

  message(toString(Sym->File) + S + Sym->getName());
}

void elf::maybeWarnUnorderableSymbol(const Symbol *Sym) {
  if (!Config->WarnSymbolOrdering)
    return;

  // If UnresolvedPolicy::Ignore is used, no "undefined symbol" error/warning
  // is emitted. It makes sense to not warn on undefined symbols.
  //
  // Note, ld.bfd --symbol-ordering-file= does not warn on undefined symbols,
  // but we don't have to be compatible here.
  if (Sym->isUndefined() &&
      Config->UnresolvedSymbols == UnresolvedPolicy::Ignore)
    return;

  const InputFile *File = Sym->File;
  auto *D = dyn_cast<Defined>(Sym);

  auto Report = [&](StringRef S) { warn(toString(File) + S + Sym->getName()); };

  if (Sym->isUndefined())
    Report(": unable to order undefined symbol: ");
  else if (Sym->isShared())
    Report(": unable to order shared symbol: ");
  else if (D && !D->Section)
    Report(": unable to order absolute symbol: ");
  else if (D && isa<OutputSection>(D->Section))
    Report(": unable to order synthetic symbol: ");
  else if (D && !D->Section->Repl->isLive())
    Report(": unable to order discarded symbol: ");
}

// Returns a symbol for an error message.
std::string lld::toString(const Symbol &B) {
  if (Config->Demangle)
    if (Optional<std::string> S = demangleItanium(B.getName()))
      return *S;
  return B.getName();
}

static uint8_t getMinVisibility(uint8_t VA, uint8_t VB) {
  if (VA == STV_DEFAULT)
    return VB;
  if (VB == STV_DEFAULT)
    return VA;
  return std::min(VA, VB);
}

// Merge symbol properties.
//
// When we have many symbols of the same name, we choose one of them,
// and that's the result of symbol resolution. However, symbols that
// were not chosen still affect some symbol properties.
void Symbol::mergeProperties(const Symbol &Other) {
  if (Other.ExportDynamic)
    ExportDynamic = true;
  if (Other.IsUsedInRegularObj)
    IsUsedInRegularObj = true;

  // DSO symbols do not affect visibility in the output.
  if (!Other.isShared())
    Visibility = getMinVisibility(Visibility, Other.Visibility);
}

void Symbol::resolve(const Symbol &Other) {
  mergeProperties(Other);

  if (isPlaceholder()) {
    replace(Other);
    return;
  }

  switch (Other.kind()) {
  case Symbol::UndefinedKind:
    resolveUndefined(cast<Undefined>(Other));
    break;
  case Symbol::CommonKind:
    resolveCommon(cast<CommonSymbol>(Other));
    break;
  case Symbol::DefinedKind:
    resolveDefined(cast<Defined>(Other));
    break;
  case Symbol::LazyArchiveKind:
    resolveLazy(cast<LazyArchive>(Other));
    break;
  case Symbol::LazyObjectKind:
    resolveLazy(cast<LazyObject>(Other));
    break;
  case Symbol::SharedKind:
    resolveShared(cast<SharedSymbol>(Other));
    break;
  case Symbol::PlaceholderKind:
    llvm_unreachable("bad symbol kind");
  }
}

void Symbol::resolveUndefined(const Undefined &Other) {
  // An undefined symbol with non default visibility must be satisfied
  // in the same DSO.
  //
  // If this is a non-weak defined symbol in a discarded section, override the
  // existing undefined symbol for better error message later.
  if ((isShared() && Other.Visibility != STV_DEFAULT) ||
      (isUndefined() && Other.Binding != STB_WEAK && Other.DiscardedSecIdx)) {
    replace(Other);
    return;
  }

  if (Traced)
    printTraceSymbol(&Other);

  if (isLazy()) {
    // An undefined weak will not fetch archive members. See comment on Lazy in
    // Symbols.h for the details.
    if (Other.Binding == STB_WEAK) {
      Binding = STB_WEAK;
      Type = Other.Type;
      return;
    }

    // Do extra check for --warn-backrefs.
    //
    // --warn-backrefs is an option to prevent an undefined reference from
    // fetching an archive member written earlier in the command line. It can be
    // used to keep compatibility with GNU linkers to some degree.
    // I'll explain the feature and why you may find it useful in this comment.
    //
    // lld's symbol resolution semantics is more relaxed than traditional Unix
    // linkers. For example,
    //
    //   ld.lld foo.a bar.o
    //
    // succeeds even if bar.o contains an undefined symbol that has to be
    // resolved by some object file in foo.a. Traditional Unix linkers don't
    // allow this kind of backward reference, as they visit each file only once
    // from left to right in the command line while resolving all undefined
    // symbols at the moment of visiting.
    //
    // In the above case, since there's no undefined symbol when a linker visits
    // foo.a, no files are pulled out from foo.a, and because the linker forgets
    // about foo.a after visiting, it can't resolve undefined symbols in bar.o
    // that could have been resolved otherwise.
    //
    // That lld accepts more relaxed form means that (besides it'd make more
    // sense) you can accidentally write a command line or a build file that
    // works only with lld, even if you have a plan to distribute it to wider
    // users who may be using GNU linkers. With --warn-backrefs, you can detect
    // a library order that doesn't work with other Unix linkers.
    //
    // The option is also useful to detect cyclic dependencies between static
    // archives. Again, lld accepts
    //
    //   ld.lld foo.a bar.a
    //
    // even if foo.a and bar.a depend on each other. With --warn-backrefs, it is
    // handled as an error.
    //
    // Here is how the option works. We assign a group ID to each file. A file
    // with a smaller group ID can pull out object files from an archive file
    // with an equal or greater group ID. Otherwise, it is a reverse dependency
    // and an error.
    //
    // A file outside --{start,end}-group gets a fresh ID when instantiated. All
    // files within the same --{start,end}-group get the same group ID. E.g.
    //
    //   ld.lld A B --start-group C D --end-group E
    //
    // A forms group 0. B form group 1. C and D (including their member object
    // files) form group 2. E forms group 3. I think that you can see how this
    // group assignment rule simulates the traditional linker's semantics.
    bool Backref = Config->WarnBackrefs && Other.File &&
                   File->GroupId < Other.File->GroupId;
    fetch();

    // We don't report backward references to weak symbols as they can be
    // overridden later.
    if (Backref && !isWeak())
      warn("backward reference detected: " + Other.getName() + " in " +
           toString(Other.File) + " refers to " + toString(File));
    return;
  }

  // Undefined symbols in a SharedFile do not change the binding.
  if (dyn_cast_or_null<SharedFile>(Other.File))
    return;

  if (isUndefined()) {
    // The binding may "upgrade" from weak to non-weak.
    if (Other.Binding != STB_WEAK)
      Binding = Other.Binding;
  } else if (auto *S = dyn_cast<SharedSymbol>(this)) {
    // The binding of a SharedSymbol will be weak if there is at least one
    // reference and all are weak. The binding has one opportunity to change to
    // weak: if the first reference is weak.
    if (Other.Binding != STB_WEAK || !S->Referenced)
      Binding = Other.Binding;
    S->Referenced = true;
  }
}

// Using .symver foo,foo@@VER unfortunately creates two symbols: foo and
// foo@@VER. We want to effectively ignore foo, so give precedence to
// foo@@VER.
// FIXME: If users can transition to using
// .symver foo,foo@@@VER
// we can delete this hack.
static int compareVersion(StringRef A, StringRef B) {
  bool X = A.contains("@@");
  bool Y = B.contains("@@");
  if (!X && Y)
    return 1;
  if (X && !Y)
    return -1;
  return 0;
}

// Compare two symbols. Return 1 if the new symbol should win, -1 if
// the new symbol should lose, or 0 if there is a conflict.
int Symbol::compare(const Symbol *Other) const {
  assert(Other->isDefined() || Other->isCommon());

  if (!isDefined() && !isCommon())
    return 1;

  if (int Cmp = compareVersion(getName(), Other->getName()))
    return Cmp;

  if (Other->isWeak())
    return -1;

  if (isWeak())
    return 1;

  if (isCommon() && Other->isCommon()) {
    if (Config->WarnCommon)
      warn("multiple common of " + getName());
    return 0;
  }

  if (isCommon()) {
    if (Config->WarnCommon)
      warn("common " + getName() + " is overridden");
    return 1;
  }

  if (Other->isCommon()) {
    if (Config->WarnCommon)
      warn("common " + getName() + " is overridden");
    return -1;
  }

  auto *OldSym = cast<Defined>(this);
  auto *NewSym = cast<Defined>(Other);

  if (Other->File && isa<BitcodeFile>(Other->File))
    return 0;

  if (!OldSym->Section && !NewSym->Section && OldSym->Value == NewSym->Value &&
      NewSym->Binding == STB_GLOBAL)
    return -1;

  return 0;
}

static void reportDuplicate(Symbol *Sym, InputFile *NewFile,
                            InputSectionBase *ErrSec, uint64_t ErrOffset) {
  if (Config->AllowMultipleDefinition)
    return;

  Defined *D = cast<Defined>(Sym);
  if (!D->Section || !ErrSec) {
    error("duplicate symbol: " + toString(*Sym) + "\n>>> defined in " +
          toString(Sym->File) + "\n>>> defined in " + toString(NewFile));
    return;
  }

  // Construct and print an error message in the form of:
  //
  //   ld.lld: error: duplicate symbol: foo
  //   >>> defined at bar.c:30
  //   >>>            bar.o (/home/alice/src/bar.o)
  //   >>> defined at baz.c:563
  //   >>>            baz.o in archive libbaz.a
  auto *Sec1 = cast<InputSectionBase>(D->Section);
  std::string Src1 = Sec1->getSrcMsg(*Sym, D->Value);
  std::string Obj1 = Sec1->getObjMsg(D->Value);
  std::string Src2 = ErrSec->getSrcMsg(*Sym, ErrOffset);
  std::string Obj2 = ErrSec->getObjMsg(ErrOffset);

  std::string Msg = "duplicate symbol: " + toString(*Sym) + "\n>>> defined at ";
  if (!Src1.empty())
    Msg += Src1 + "\n>>>            ";
  Msg += Obj1 + "\n>>> defined at ";
  if (!Src2.empty())
    Msg += Src2 + "\n>>>            ";
  Msg += Obj2;
  error(Msg);
}

void Symbol::resolveCommon(const CommonSymbol &Other) {
  int Cmp = compare(&Other);
  if (Cmp < 0)
    return;

  if (Cmp > 0) {
    replace(Other);
    return;
  }

  CommonSymbol *OldSym = cast<CommonSymbol>(this);

  OldSym->Alignment = std::max(OldSym->Alignment, Other.Alignment);
  if (OldSym->Size < Other.Size) {
    OldSym->File = Other.File;
    OldSym->Size = Other.Size;
  }
}

void Symbol::resolveDefined(const Defined &Other) {
  int Cmp = compare(&Other);
  if (Cmp > 0)
    replace(Other);
  else if (Cmp == 0)
    reportDuplicate(this, Other.File,
                    dyn_cast_or_null<InputSectionBase>(Other.Section),
                    Other.Value);
}

template <class LazyT> void Symbol::resolveLazy(const LazyT &Other) {
  if (!isUndefined())
    return;

  // An undefined weak will not fetch archive members. See comment on Lazy in
  // Symbols.h for the details.
  if (isWeak()) {
    uint8_t Ty = Type;
    replace(Other);
    Type = Ty;
    Binding = STB_WEAK;
    return;
  }

  Other.fetch();
}

void Symbol::resolveShared(const SharedSymbol &Other) {
  if (Visibility == STV_DEFAULT && (isUndefined() || isLazy())) {
    // An undefined symbol with non default visibility must be satisfied
    // in the same DSO.
    uint8_t Bind = Binding;
    replace(Other);
    Binding = Bind;
    cast<SharedSymbol>(this)->Referenced = true;
  }
}
