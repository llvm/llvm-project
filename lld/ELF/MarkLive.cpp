//===- MarkLive.cpp -------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements --gc-sections, which is a feature to remove unused
// sections from output. Unused sections are sections that are not reachable
// from known GC-root symbols or sections. Naturally the feature is
// implemented as a mark-sweep garbage collector.
//
// Here's how it works. Each InputSectionBase has a "Live" bit. The bit is off
// by default. Starting with GC-root symbols or sections, markLive function
// defined in this file visits all reachable sections to set their Live
// bits. Writer will then ignore sections whose Live bits are off, so that
// such sections are not included into output.
//
//===----------------------------------------------------------------------===//

#include "InputSection.h"
#include "LinkerScript.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "Target.h"
#include "Writer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELF.h"
#include <functional>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

using namespace lld;
using namespace lld::elf;

// A resolved relocation. The Sec and Offset fields are set if the relocation
// was resolved to an offset within a section.
template <class ELFT>
struct ResolvedReloc {
  InputSectionBase<ELFT> *Sec;
  typename ELFT::uint Offset;
};

template <class ELFT>
static typename ELFT::uint getAddend(InputSectionBase<ELFT> *Sec,
                                     const typename ELFT::Rel &Rel) {
  return Target->getImplicitAddend(Sec->getSectionData().begin(),
                                   Rel.getType(Config->Mips64EL));
}

template <class ELFT>
static typename ELFT::uint getAddend(InputSectionBase<ELFT> *Sec,
                                     const typename ELFT::Rela &Rel) {
  return Rel.r_addend;
}

template <class ELFT, class RelT>
static ResolvedReloc<ELFT> resolveReloc(InputSection<ELFT> *Sec, RelT &Rel) {
  SymbolBody &B = Sec->getFile()->getRelocTargetSym(Rel);
  auto *D = dyn_cast<DefinedRegular<ELFT>>(&B);
  if (!D || !D->Section)
    return {nullptr, 0};
  typename ELFT::uint Offset = D->Value;
  if (D->isSection())
    Offset += getAddend(Sec, Rel);
  return {D->Section->Repl, Offset};
}

// Calls Fn for each section that Sec refers to via relocations.
template <class ELFT>
static void forEachSuccessor(InputSection<ELFT> *Sec,
                             std::function<void(ResolvedReloc<ELFT>)> Fn) {
  typedef typename ELFT::Rel Elf_Rel;
  typedef typename ELFT::Rela Elf_Rela;
  typedef typename ELFT::Shdr Elf_Shdr;

  ELFFile<ELFT> &Obj = Sec->getFile()->getObj();
  for (const Elf_Shdr *RelSec : Sec->RelocSections) {
    if (RelSec->sh_type == SHT_RELA) {
      for (const Elf_Rela &RI : Obj.relas(RelSec))
        Fn(resolveReloc(Sec, RI));
    } else {
      for (const Elf_Rel &RI : Obj.rels(RelSec))
        Fn(resolveReloc(Sec, RI));
    }
  }
}

// Sections listed below are special because they are used by the loader
// just by being in an ELF file. They should not be garbage-collected.
template <class ELFT> static bool isReserved(InputSectionBase<ELFT> *Sec) {
  switch (Sec->getSectionHdr()->sh_type) {
  case SHT_FINI_ARRAY:
  case SHT_INIT_ARRAY:
  case SHT_NOTE:
  case SHT_PREINIT_ARRAY:
    return true;
  default:
    StringRef S = Sec->getSectionName();

    // We do not want to reclaim sections if they can be referred
    // by __start_* and __stop_* symbols.
    if (isValidCIdentifier(S))
      return true;

    return S.startswith(".ctors") || S.startswith(".dtors") ||
           S.startswith(".init") || S.startswith(".fini") ||
           S.startswith(".jcr");
  }
}

// This is the main function of the garbage collector.
// Starting from GC-root sections, this function visits all reachable
// sections to set their "Live" bits.
template <class ELFT> void elf::markLive(SymbolTable<ELFT> *Symtab) {
  typedef typename ELFT::uint uintX_t;
  SmallVector<InputSection<ELFT> *, 256> Q;

  auto Enqueue = [&](ResolvedReloc<ELFT> R) {
    if (!R.Sec)
      return;
    if (auto *MS = dyn_cast<MergeInputSection<ELFT>>(R.Sec)) {
      std::pair<std::pair<uintX_t, uintX_t> *, uintX_t> T =
          MS->getRangeAndSize(R.Offset);
      T.first->second = 0;
    }
    if (R.Sec->Live)
      return;
    R.Sec->Live = true;
    if (InputSection<ELFT> *S = dyn_cast<InputSection<ELFT>>(R.Sec))
      Q.push_back(S);
  };

  auto MarkSymbol = [&](SymbolBody *Sym) {
    if (Sym)
      if (auto *D = dyn_cast<DefinedRegular<ELFT>>(Sym))
        Enqueue({D->Section, D->Value});
  };

  // Add GC root symbols.
  if (Config->EntrySym)
    MarkSymbol(Config->EntrySym->Body);
  MarkSymbol(Symtab->find(Config->Init));
  MarkSymbol(Symtab->find(Config->Fini));
  for (StringRef S : Config->Undefined)
    MarkSymbol(Symtab->find(S));

  // Preserve externally-visible symbols if the symbols defined by this
  // file can interrupt other ELF file's symbols at runtime.
  for (const Symbol *S : Symtab->getSymbols())
    if (S->includeInDynsym())
      MarkSymbol(S->Body);

  // Preserve special sections and those which are specified in linker
  // script KEEP command.
  for (const std::unique_ptr<ObjectFile<ELFT>> &F : Symtab->getObjectFiles())
    for (InputSectionBase<ELFT> *Sec : F->getSections())
      if (Sec && Sec != &InputSection<ELFT>::Discarded)
        if (isReserved(Sec) || Script<ELFT>::X->shouldKeep(Sec))
          Enqueue({Sec, 0});

  // Mark all reachable sections.
  while (!Q.empty())
    forEachSuccessor<ELFT>(Q.pop_back_val(), Enqueue);
}

template void elf::markLive<ELF32LE>(SymbolTable<ELF32LE> *);
template void elf::markLive<ELF32BE>(SymbolTable<ELF32BE> *);
template void elf::markLive<ELF64LE>(SymbolTable<ELF64LE> *);
template void elf::markLive<ELF64BE>(SymbolTable<ELF64BE> *);
