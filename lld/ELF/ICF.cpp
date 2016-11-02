//===- ICF.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Identical Code Folding is a feature to merge sections not by name (which
// is regular comdat handling) but by contents. If two non-writable sections
// have the same data, relocations, attributes, etc., then the two
// are considered identical and merged by the linker. This optimization
// makes outputs smaller.
//
// ICF is theoretically a problem of reducing graphs by merging as many
// identical subgraphs as possible if we consider sections as vertices and
// relocations as edges. It may sound simple, but it is a bit more
// complicated than you might think. The order of processing sections
// matters because merging two sections can make other sections, whose
// relocations now point to the same section, mergeable. Graphs may contain
// cycles. We need a sophisticated algorithm to do this properly and
// efficiently.
//
// What we do in this file is this. We split sections into groups. Sections
// in the same group are considered identical.
//
// We begin by optimistically putting all sections into a single equivalence
// class. Then we apply a series of checks that split this initial
// equivalence class into more and more refined equivalence classes based on
// the properties by which a section can be distinguished.
//
// We begin by checking that the section contents and flags are the
// same. This only needs to be done once since these properties don't depend
// on the current equivalence class assignment.
//
// Then we split the equivalence classes based on checking that their
// relocations are the same, where relocation targets are compared by their
// equivalence class, not the concrete section. This may need to be done
// multiple times because as the equivalence classes are refined, two
// sections that had a relocation target in the same equivalence class may
// now target different equivalence classes, and hence these two sections
// must be put in different equivalence classes (whereas in the previous
// iteration they were not since the relocation target was the same.)
//
// Our algorithm is smart enough to merge the following mutually-recursive
// functions.
//
//   void foo() { bar(); }
//   void bar() { foo(); }
//
// This algorithm is so-called "optimistic" algorithm described in
// http://research.google.com/pubs/pub36912.html. (Note that what GNU
// gold implemented is different from the optimistic algorithm.)
//
//===----------------------------------------------------------------------===//

#include "ICF.h"
#include "Config.h"
#include "OutputSections.h"
#include "SymbolTable.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/raw_ostream.h"

using namespace lld;
using namespace lld::elf;
using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

namespace lld {
namespace elf {
template <class ELFT> class ICF {
  typedef typename ELFT::Shdr Elf_Shdr;
  typedef typename ELFT::Sym Elf_Sym;
  typedef typename ELFT::uint uintX_t;
  typedef Elf_Rel_Impl<ELFT, false> Elf_Rel;

  using Comparator = std::function<bool(const InputSection<ELFT> *,
                                        const InputSection<ELFT> *)>;

public:
  void run();

private:
  uint64_t NextId = 1;

  static void setLive(SymbolTable<ELFT> *S);
  static uint64_t relSize(InputSection<ELFT> *S);
  static uint64_t getHash(InputSection<ELFT> *S);
  static bool isEligible(InputSectionBase<ELFT> *Sec);
  static std::vector<InputSection<ELFT> *> getSections();

  void segregate(InputSection<ELFT> **Begin, InputSection<ELFT> **End,
                 Comparator Eq);

  void forEachGroup(std::vector<InputSection<ELFT> *> &V, Comparator Eq);

  template <class RelTy>
  static bool relocationEq(ArrayRef<RelTy> RA, ArrayRef<RelTy> RB);

  template <class RelTy>
  static bool variableEq(const InputSection<ELFT> *A,
                         const InputSection<ELFT> *B, ArrayRef<RelTy> RA,
                         ArrayRef<RelTy> RB);

  static bool equalsConstant(const InputSection<ELFT> *A,
                             const InputSection<ELFT> *B);

  static bool equalsVariable(const InputSection<ELFT> *A,
                             const InputSection<ELFT> *B);
};
}
}

// Returns a hash value for S. Note that the information about
// relocation targets is not included in the hash value.
template <class ELFT> uint64_t ICF<ELFT>::getHash(InputSection<ELFT> *S) {
  uint64_t Flags = S->getSectionHdr()->sh_flags;
  uint64_t H = hash_combine(Flags, S->getSize());
  for (const Elf_Shdr *Rel : S->RelocSections)
    H = hash_combine(H, (uint64_t)Rel->sh_size);
  return H;
}

// Returns true if Sec is subject of ICF.
template <class ELFT> bool ICF<ELFT>::isEligible(InputSectionBase<ELFT> *Sec) {
  if (!Sec || Sec == &InputSection<ELFT>::Discarded || !Sec->Live)
    return false;
  auto *S = dyn_cast<InputSection<ELFT>>(Sec);
  if (!S)
    return false;

  // .init and .fini contains instructions that must be executed to
  // initialize and finalize the process. They cannot and should not
  // be merged.
  StringRef Name = S->Name;
  if (Name == ".init" || Name == ".fini")
    return false;

  const Elf_Shdr &H = *S->getSectionHdr();
  return (H.sh_flags & SHF_ALLOC) && (~H.sh_flags & SHF_WRITE);
}

template <class ELFT>
std::vector<InputSection<ELFT> *> ICF<ELFT>::getSections() {
  std::vector<InputSection<ELFT> *> V;
  for (ObjectFile<ELFT> *F : Symtab<ELFT>::X->getObjectFiles())
    for (InputSectionBase<ELFT> *S : F->getSections())
      if (isEligible(S))
        V.push_back(cast<InputSection<ELFT>>(S));
  return V;
}

// All sections between Begin and End must have the same group ID before
// you call this function. This function compare sections between Begin
// and End using Eq and assign new group IDs for new groups.
template <class ELFT>
void ICF<ELFT>::segregate(InputSection<ELFT> **Begin, InputSection<ELFT> **End,
                          Comparator Eq) {
  // This loop rearranges [Begin, End) so that all sections that are
  // equal in terms of Eq are contiguous. The algorithm is quadratic in
  // the worst case, but that is not an issue in practice because the
  // number of distinct sections in [Begin, End) is usually very small.
  InputSection<ELFT> **I = Begin;
  for (;;) {
    InputSection<ELFT> *Head = *I;
    auto Bound = std::stable_partition(
        I + 1, End, [&](InputSection<ELFT> *S) { return Eq(Head, S); });
    if (Bound == End)
      return;
    uint64_t Id = NextId++;
    for (; I != Bound; ++I)
      (*I)->GroupId = Id;
  }
}

template <class ELFT>
void ICF<ELFT>::forEachGroup(std::vector<InputSection<ELFT> *> &V,
                             Comparator Eq) {
  for (InputSection<ELFT> **I = V.data(), **E = I + V.size(); I != E;) {
    InputSection<ELFT> *Head = *I;
    auto Bound = std::find_if(I + 1, E, [&](InputSection<ELFT> *S) {
      return S->GroupId != Head->GroupId;
    });
    segregate(I, Bound, Eq);
    I = Bound;
  }
}

// Compare two lists of relocations.
template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::relocationEq(ArrayRef<RelTy> RelsA, ArrayRef<RelTy> RelsB) {
  const RelTy *IA = RelsA.begin();
  const RelTy *EA = RelsA.end();
  const RelTy *IB = RelsB.begin();
  const RelTy *EB = RelsB.end();
  if (EA - IA != EB - IB)
    return false;
  for (; IA != EA; ++IA, ++IB)
    if (IA->r_offset != IB->r_offset ||
        IA->getType(Config->Mips64EL) != IB->getType(Config->Mips64EL) ||
        getAddend<ELFT>(*IA) != getAddend<ELFT>(*IB))
      return false;
  return true;
}

// Compare "non-moving" part of two InputSections, namely everything
// except relocation targets.
template <class ELFT>
bool ICF<ELFT>::equalsConstant(const InputSection<ELFT> *A,
                               const InputSection<ELFT> *B) {
  if (A->RelocSections.size() != B->RelocSections.size())
    return false;

  for (size_t I = 0, E = A->RelocSections.size(); I != E; ++I) {
    const Elf_Shdr *RA = A->RelocSections[I];
    const Elf_Shdr *RB = B->RelocSections[I];
    ELFFile<ELFT> &FileA = A->File->getObj();
    ELFFile<ELFT> &FileB = B->File->getObj();
    if (RA->sh_type == SHT_RELA) {
      if (!relocationEq(FileA.relas(RA), FileB.relas(RB)))
        return false;
    } else {
      if (!relocationEq(FileA.rels(RA), FileB.rels(RB)))
        return false;
    }
  }

  return A->getSectionHdr()->sh_flags == B->getSectionHdr()->sh_flags &&
         A->getSize() == B->getSize() && A->Data == B->Data;
}

template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::variableEq(const InputSection<ELFT> *A,
                           const InputSection<ELFT> *B, ArrayRef<RelTy> RelsA,
                           ArrayRef<RelTy> RelsB) {
  const RelTy *IA = RelsA.begin();
  const RelTy *EA = RelsA.end();
  const RelTy *IB = RelsB.begin();
  for (; IA != EA; ++IA, ++IB) {
    SymbolBody &SA = A->File->getRelocTargetSym(*IA);
    SymbolBody &SB = B->File->getRelocTargetSym(*IB);
    if (&SA == &SB)
      continue;

    // Or, the symbols should be pointing to the same section
    // in terms of the group ID.
    auto *DA = dyn_cast<DefinedRegular<ELFT>>(&SA);
    auto *DB = dyn_cast<DefinedRegular<ELFT>>(&SB);
    if (!DA || !DB)
      return false;
    if (DA->Value != DB->Value)
      return false;
    InputSection<ELFT> *X = dyn_cast<InputSection<ELFT>>(DA->Section);
    InputSection<ELFT> *Y = dyn_cast<InputSection<ELFT>>(DB->Section);
    if (X && Y && X->GroupId && X->GroupId == Y->GroupId)
      continue;
    return false;
  }
  return true;
}

// Compare "moving" part of two InputSections, namely relocation targets.
template <class ELFT>
bool ICF<ELFT>::equalsVariable(const InputSection<ELFT> *A,
                               const InputSection<ELFT> *B) {
  for (size_t I = 0, E = A->RelocSections.size(); I != E; ++I) {
    const Elf_Shdr *RA = A->RelocSections[I];
    const Elf_Shdr *RB = B->RelocSections[I];
    ELFFile<ELFT> &FileA = A->File->getObj();
    ELFFile<ELFT> &FileB = B->File->getObj();
    if (RA->sh_type == SHT_RELA) {
      if (!variableEq(A, B, FileA.relas(RA), FileB.relas(RB)))
        return false;
    } else {
      if (!variableEq(A, B, FileA.rels(RA), FileB.rels(RB)))
        return false;
    }
  }
  return true;
}

// The main function of ICF.
template <class ELFT> void ICF<ELFT>::run() {
  // Initially, we use hash values as section group IDs. Therefore,
  // if two sections have the same ID, they are likely (but not
  // guaranteed) to have the same static contents in terms of ICF.
  std::vector<InputSection<ELFT> *> V = getSections();
  for (InputSection<ELFT> *S : V)
    // Set MSB on to avoid collisions with serial group IDs
    S->GroupId = getHash(S) | (uint64_t(1) << 63);

  // From now on, sections in V are ordered so that sections in
  // the same group are consecutive in the vector.
  std::stable_sort(V.begin(), V.end(),
                   [](InputSection<ELFT> *A, InputSection<ELFT> *B) {
                     if (A->GroupId != B->GroupId)
                       return A->GroupId < B->GroupId;
                     // Within a group, put the highest alignment
                     // requirement first, so that's the one we'll keep.
                     return B->Alignment < A->Alignment;
                   });

  // Compare static contents and assign unique IDs for each static content.
  forEachGroup(V, equalsConstant);

  // Split groups by comparing relocations until we get a convergence.
  int Cnt = 1;
  for (;;) {
    ++Cnt;
    uint64_t Id = NextId;
    forEachGroup(V, equalsVariable);
    if (Id == NextId)
      break;
  }
  log("ICF needed " + Twine(Cnt) + " iterations.");

  // Merge sections in the same group.
  for (auto I = V.begin(), E = V.end(); I != E;) {
    InputSection<ELFT> *Head = *I++;
    auto Bound = std::find_if(I, E, [&](InputSection<ELFT> *S) {
      return Head->GroupId != S->GroupId;
    });
    if (I == Bound)
      continue;
    log("selected " + Head->Name);
    while (I != Bound) {
      InputSection<ELFT> *S = *I++;
      log("  removed " + S->Name);
      Head->replace(S);
    }
  }
}

// ICF entry point function.
template <class ELFT> void elf::doIcf() { ICF<ELFT>().run(); }

template void elf::doIcf<ELF32LE>();
template void elf::doIcf<ELF32BE>();
template void elf::doIcf<ELF64LE>();
template void elf::doIcf<ELF64BE>();
