//===- MarkLive.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

#include "MarkLive.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/Memory.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Object/ELF.h"
#include <functional>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;

using namespace lld;
using namespace lld::elf;

namespace {
template <class ELFT> class MarkLive {
public:
  MarkLive(unsigned Partition) : Partition(Partition) {}

  void run();
  void moveToMain();

private:
  void enqueue(InputSectionBase *Sec, uint64_t Offset);
  void markSymbol(Symbol *Sym);
  void mark();

  template <class RelTy>
  void resolveReloc(InputSectionBase &Sec, RelTy &Rel, bool IsLSDA);

  template <class RelTy>
  void scanEhFrameSection(EhInputSection &EH, ArrayRef<RelTy> Rels);

  // The index of the partition that we are currently processing.
  unsigned Partition;

  // A list of sections to visit.
  SmallVector<InputSection *, 256> Queue;

  // There are normally few input sections whose names are valid C
  // identifiers, so we just store a std::vector instead of a multimap.
  DenseMap<StringRef, std::vector<InputSectionBase *>> CNamedSections;
};
} // namespace

template <class ELFT>
static uint64_t getAddend(InputSectionBase &Sec,
                          const typename ELFT::Rel &Rel) {
  return Target->getImplicitAddend(Sec.data().begin() + Rel.r_offset,
                                   Rel.getType(Config->IsMips64EL));
}

template <class ELFT>
static uint64_t getAddend(InputSectionBase &Sec,
                          const typename ELFT::Rela &Rel) {
  return Rel.r_addend;
}

template <class ELFT>
template <class RelTy>
void MarkLive<ELFT>::resolveReloc(InputSectionBase &Sec, RelTy &Rel,
                                  bool IsLSDA) {
  Symbol &Sym = Sec.getFile<ELFT>()->getRelocTargetSym(Rel);

  // If a symbol is referenced in a live section, it is used.
  Sym.Used = true;

  if (auto *D = dyn_cast<Defined>(&Sym)) {
    auto *RelSec = dyn_cast_or_null<InputSectionBase>(D->Section);
    if (!RelSec)
      return;

    uint64_t Offset = D->Value;
    if (D->isSection())
      Offset += getAddend<ELFT>(Sec, Rel);

    if (!IsLSDA || !(RelSec->Flags & SHF_EXECINSTR))
      enqueue(RelSec, Offset);
    return;
  }

  if (auto *SS = dyn_cast<SharedSymbol>(&Sym))
    if (!SS->isWeak())
      SS->getFile().IsNeeded = true;

  for (InputSectionBase *Sec : CNamedSections.lookup(Sym.getName()))
    enqueue(Sec, 0);
}

// The .eh_frame section is an unfortunate special case.
// The section is divided in CIEs and FDEs and the relocations it can have are
// * CIEs can refer to a personality function.
// * FDEs can refer to a LSDA
// * FDEs refer to the function they contain information about
// The last kind of relocation cannot keep the referred section alive, or they
// would keep everything alive in a common object file. In fact, each FDE is
// alive if the section it refers to is alive.
// To keep things simple, in here we just ignore the last relocation kind. The
// other two keep the referred section alive.
//
// A possible improvement would be to fully process .eh_frame in the middle of
// the gc pass. With that we would be able to also gc some sections holding
// LSDAs and personality functions if we found that they were unused.
template <class ELFT>
template <class RelTy>
void MarkLive<ELFT>::scanEhFrameSection(EhInputSection &EH,
                                        ArrayRef<RelTy> Rels) {
  for (size_t I = 0, End = EH.Pieces.size(); I < End; ++I) {
    EhSectionPiece &Piece = EH.Pieces[I];
    size_t FirstRelI = Piece.FirstRelocation;
    if (FirstRelI == (unsigned)-1)
      continue;

    if (read32<ELFT::TargetEndianness>(Piece.data().data() + 4) == 0) {
      // This is a CIE, we only need to worry about the first relocation. It is
      // known to point to the personality function.
      resolveReloc(EH, Rels[FirstRelI], false);
      continue;
    }

    // This is a FDE. The relocations point to the described function or to
    // a LSDA. We only need to keep the LSDA alive, so ignore anything that
    // points to executable sections.
    uint64_t PieceEnd = Piece.InputOff + Piece.Size;
    for (size_t J = FirstRelI, End2 = Rels.size(); J < End2; ++J)
      if (Rels[J].r_offset < PieceEnd)
        resolveReloc(EH, Rels[J], true);
  }
}

// Some sections are used directly by the loader, so they should never be
// garbage-collected. This function returns true if a given section is such
// section.
static bool isReserved(InputSectionBase *Sec) {
  switch (Sec->Type) {
  case SHT_FINI_ARRAY:
  case SHT_INIT_ARRAY:
  case SHT_NOTE:
  case SHT_PREINIT_ARRAY:
    return true;
  default:
    StringRef S = Sec->Name;
    return S.startswith(".ctors") || S.startswith(".dtors") ||
           S.startswith(".init") || S.startswith(".fini") ||
           S.startswith(".jcr");
  }
}

template <class ELFT>
void MarkLive<ELFT>::enqueue(InputSectionBase *Sec, uint64_t Offset) {
  // Skip over discarded sections. This in theory shouldn't happen, because
  // the ELF spec doesn't allow a relocation to point to a deduplicated
  // COMDAT section directly. Unfortunately this happens in practice (e.g.
  // .eh_frame) so we need to add a check.
  if (Sec == &InputSection::Discarded)
    return;

  // Usually, a whole section is marked as live or dead, but in mergeable
  // (splittable) sections, each piece of data has independent liveness bit.
  // So we explicitly tell it which offset is in use.
  if (auto *MS = dyn_cast<MergeInputSection>(Sec))
    MS->getSectionPiece(Offset)->Live = true;

  // Set Sec->Partition to the meet (i.e. the "minimum") of Partition and
  // Sec->Partition in the following lattice: 1 < other < 0. If Sec->Partition
  // doesn't change, we don't need to do anything.
  if (Sec->Partition == 1 || Sec->Partition == Partition)
    return;
  Sec->Partition = Sec->Partition ? 1 : Partition;

  // Add input section to the queue.
  if (InputSection *S = dyn_cast<InputSection>(Sec))
    Queue.push_back(S);
}

template <class ELFT> void MarkLive<ELFT>::markSymbol(Symbol *Sym) {
  if (auto *D = dyn_cast_or_null<Defined>(Sym))
    if (auto *IS = dyn_cast_or_null<InputSectionBase>(D->Section))
      enqueue(IS, D->Value);
}

// This is the main function of the garbage collector.
// Starting from GC-root sections, this function visits all reachable
// sections to set their "Live" bits.
template <class ELFT> void MarkLive<ELFT>::run() {
  // Add GC root symbols.

  // Preserve externally-visible symbols if the symbols defined by this
  // file can interrupt other ELF file's symbols at runtime.
  Symtab->forEachSymbol([&](Symbol *Sym) {
    if (Sym->includeInDynsym() && Sym->Partition == Partition)
      markSymbol(Sym);
  });

  // If this isn't the main partition, that's all that we need to preserve.
  if (Partition != 1) {
    mark();
    return;
  }

  markSymbol(Symtab->find(Config->Entry));
  markSymbol(Symtab->find(Config->Init));
  markSymbol(Symtab->find(Config->Fini));
  for (StringRef S : Config->Undefined)
    markSymbol(Symtab->find(S));
  for (StringRef S : Script->ReferencedSymbols)
    markSymbol(Symtab->find(S));

  // Preserve special sections and those which are specified in linker
  // script KEEP command.
  for (InputSectionBase *Sec : InputSections) {
    // Mark .eh_frame sections as live because there are usually no relocations
    // that point to .eh_frames. Otherwise, the garbage collector would drop
    // all of them. We also want to preserve personality routines and LSDA
    // referenced by .eh_frame sections, so we scan them for that here.
    if (auto *EH = dyn_cast<EhInputSection>(Sec)) {
      EH->markLive();
      if (!EH->NumRelocations)
        continue;

      if (EH->AreRelocsRela)
        scanEhFrameSection(*EH, EH->template relas<ELFT>());
      else
        scanEhFrameSection(*EH, EH->template rels<ELFT>());
    }

    if (Sec->Flags & SHF_LINK_ORDER)
      continue;

    if (isReserved(Sec) || Script->shouldKeep(Sec)) {
      enqueue(Sec, 0);
    } else if (isValidCIdentifier(Sec->Name)) {
      CNamedSections[Saver.save("__start_" + Sec->Name)].push_back(Sec);
      CNamedSections[Saver.save("__stop_" + Sec->Name)].push_back(Sec);
    }
  }

  mark();
}

template <class ELFT> void MarkLive<ELFT>::mark() {
  // Mark all reachable sections.
  while (!Queue.empty()) {
    InputSectionBase &Sec = *Queue.pop_back_val();

    if (Sec.AreRelocsRela) {
      for (const typename ELFT::Rela &Rel : Sec.template relas<ELFT>())
        resolveReloc(Sec, Rel, false);
    } else {
      for (const typename ELFT::Rel &Rel : Sec.template rels<ELFT>())
        resolveReloc(Sec, Rel, false);
    }

    for (InputSectionBase *IS : Sec.DependentSections)
      enqueue(IS, 0);
  }
}

// Move the sections for some symbols to the main partition, specifically ifuncs
// (because they can result in an IRELATIVE being added to the main partition's
// GOT, which means that the ifunc must be available when the main partition is
// loaded) and TLS symbols (because we only know how to correctly process TLS
// relocations for the main partition).
template <class ELFT> void MarkLive<ELFT>::moveToMain() {
  for (InputFile *File : ObjectFiles)
    for (Symbol *S : File->getSymbols())
      if (auto *D = dyn_cast<Defined>(S))
        if ((D->Type == STT_GNU_IFUNC || D->Type == STT_TLS) && D->Section &&
            D->Section->isLive())
          markSymbol(S);

  mark();
}

// Before calling this function, Live bits are off for all
// input sections. This function make some or all of them on
// so that they are emitted to the output file.
template <class ELFT> void elf::markLive() {
  // If -gc-sections is not given, no sections are removed.
  if (!Config->GcSections) {
    for (InputSectionBase *Sec : InputSections)
      Sec->markLive();

    // If a DSO defines a symbol referenced in a regular object, it is needed.
    Symtab->forEachSymbol([](Symbol *Sym) {
      if (auto *S = dyn_cast<SharedSymbol>(Sym))
        if (S->IsUsedInRegularObj && !S->isWeak())
          S->getFile().IsNeeded = true;
    });
    return;
  }

  // Otheriwse, do mark-sweep GC.
  //
  // The -gc-sections option works only for SHF_ALLOC sections
  // (sections that are memory-mapped at runtime). So we can
  // unconditionally make non-SHF_ALLOC sections alive except
  // SHF_LINK_ORDER and SHT_REL/SHT_RELA sections.
  //
  // Usually, non-SHF_ALLOC sections are not removed even if they are
  // unreachable through relocations because reachability is not
  // a good signal whether they are garbage or not (e.g. there is
  // usually no section referring to a .comment section, but we
  // want to keep it.).
  //
  // Note on SHF_LINK_ORDER: Such sections contain metadata and they
  // have a reverse dependency on the InputSection they are linked with.
  // We are able to garbage collect them.
  //
  // Note on SHF_REL{,A}: Such sections reach here only when -r
  // or -emit-reloc were given. And they are subject of garbage
  // collection because, if we remove a text section, we also
  // remove its relocation section.
  for (InputSectionBase *Sec : InputSections) {
    bool IsAlloc = (Sec->Flags & SHF_ALLOC);
    bool IsLinkOrder = (Sec->Flags & SHF_LINK_ORDER);
    bool IsRel = (Sec->Type == SHT_REL || Sec->Type == SHT_RELA);

    if (!IsAlloc && !IsLinkOrder && !IsRel)
      Sec->markLive();
  }

  // Follow the graph to mark all live sections.
  for (unsigned CurPart = 1; CurPart <= Partitions.size(); ++CurPart)
    MarkLive<ELFT>(CurPart).run();

  // If we have multiple partitions, some sections need to live in the main
  // partition even if they were allocated to a loadable partition. Move them
  // there now.
  if (Partitions.size() != 1)
    MarkLive<ELFT>(1).moveToMain();

  // Report garbage-collected sections.
  if (Config->PrintGcSections)
    for (InputSectionBase *Sec : InputSections)
      if (!Sec->isLive())
        message("removing unused section " + toString(Sec));
}

template void elf::markLive<ELF32LE>();
template void elf::markLive<ELF32BE>();
template void elf::markLive<ELF64LE>();
template void elf::markLive<ELF64BE>();
