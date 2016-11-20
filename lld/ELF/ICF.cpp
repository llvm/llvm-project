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
#include "SymbolTable.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include <algorithm>

using namespace lld;
using namespace lld::elf;
using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;

namespace {
template <class ELFT> class ICF {
public:
  void run();

private:
  uint64_t NextId = 1;

  using Comparator = std::function<bool(const InputSection<ELFT> *,
                                        const InputSection<ELFT> *)>;

  void segregate(MutableArrayRef<InputSection<ELFT> *> Arr, Comparator Eq);

  void
  forEachGroup(std::vector<InputSection<ELFT> *> &V,
               std::function<void(MutableArrayRef<InputSection<ELFT> *>)> Fn);
};
}

// Returns a hash value for S. Note that the information about
// relocation targets is not included in the hash value.
template <class ELFT> static uint64_t getHash(InputSection<ELFT> *S) {
  return hash_combine(S->Flags, S->getSize(), S->NumRelocations);
}

// Returns true if section S is subject of ICF.
template <class ELFT> static bool isEligible(InputSection<ELFT> *S) {
  // .init and .fini contains instructions that must be executed to
  // initialize and finalize the process. They cannot and should not
  // be merged.
  return S->Live && (S->Flags & SHF_ALLOC) && !(S->Flags & SHF_WRITE) &&
         S->Name != ".init" && S->Name != ".fini";
}

template <class ELFT> static std::vector<InputSection<ELFT> *> getSections() {
  std::vector<InputSection<ELFT> *> V;
  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections)
    if (auto *S = dyn_cast<InputSection<ELFT>>(Sec))
      if (isEligible(S))
        V.push_back(S);
  return V;
}

// All sections between Begin and End must have the same group ID before
// you call this function. This function compare sections between Begin
// and End using Eq and assign new group IDs for new groups.
template <class ELFT>
void ICF<ELFT>::segregate(MutableArrayRef<InputSection<ELFT> *> Arr,
                          Comparator Eq) {
  // This loop rearranges [Begin, End) so that all sections that are
  // equal in terms of Eq are contiguous. The algorithm is quadratic in
  // the worst case, but that is not an issue in practice because the
  // number of distinct sections in [Begin, End) is usually very small.
  InputSection<ELFT> **I = Arr.begin();
  for (;;) {
    InputSection<ELFT> *Head = *I;
    auto Bound = std::stable_partition(
        I + 1, Arr.end(), [&](InputSection<ELFT> *S) { return Eq(Head, S); });
    if (Bound == Arr.end())
      return;
    uint64_t Id = NextId++;
    for (; I != Bound; ++I)
      (*I)->GroupId = Id;
  }
}

template <class ELFT>
void ICF<ELFT>::forEachGroup(
    std::vector<InputSection<ELFT> *> &V,
    std::function<void(MutableArrayRef<InputSection<ELFT> *>)> Fn) {
  for (InputSection<ELFT> **I = V.data(), **E = I + V.size(); I != E;) {
    InputSection<ELFT> *Head = *I;
    auto Bound = std::find_if(I + 1, E, [&](InputSection<ELFT> *S) {
      return S->GroupId != Head->GroupId;
    });
    Fn({I, Bound});
    I = Bound;
  }
}

// Compare two lists of relocations.
template <class ELFT, class RelTy>
static bool relocationEq(ArrayRef<RelTy> RelsA, ArrayRef<RelTy> RelsB) {
  auto Eq = [](const RelTy &A, const RelTy &B) {
    return A.r_offset == B.r_offset &&
           A.getType(Config->Mips64EL) == B.getType(Config->Mips64EL) &&
           getAddend<ELFT>(A) == getAddend<ELFT>(B);
  };

  return RelsA.size() == RelsB.size() &&
         std::equal(RelsA.begin(), RelsA.end(), RelsB.begin(), Eq);
}

// Compare "non-moving" part of two InputSections, namely everything
// except relocation targets.
template <class ELFT>
static bool equalsConstant(const InputSection<ELFT> *A,
                           const InputSection<ELFT> *B) {
  if (A->NumRelocations != B->NumRelocations || A->Flags != B->Flags ||
      A->getSize() != B->getSize() || A->Data != B->Data)
    return false;

  if (A->AreRelocsRela)
    return relocationEq<ELFT>(A->relas(), B->relas());
  return relocationEq<ELFT>(A->rels(), B->rels());
}

template <class ELFT, class RelTy>
static bool variableEq(const InputSection<ELFT> *A, ArrayRef<RelTy> RelsA,
                       const InputSection<ELFT> *B, ArrayRef<RelTy> RelsB) {
  auto Eq = [&](const RelTy &RA, const RelTy &RB) {
    SymbolBody &SA = A->getFile()->getRelocTargetSym(RA);
    SymbolBody &SB = B->getFile()->getRelocTargetSym(RB);
    if (&SA == &SB)
      return true;

    // Or, the symbols should be pointing to the same section
    // in terms of the group ID.
    auto *DA = dyn_cast<DefinedRegular<ELFT>>(&SA);
    auto *DB = dyn_cast<DefinedRegular<ELFT>>(&SB);
    if (!DA || !DB)
      return false;
    if (DA->Value != DB->Value)
      return false;
    auto *X = dyn_cast<InputSection<ELFT>>(DA->Section);
    auto *Y = dyn_cast<InputSection<ELFT>>(DB->Section);
    return X && Y && X->GroupId && X->GroupId == Y->GroupId;
  };

  return std::equal(RelsA.begin(), RelsA.end(), RelsB.begin(), Eq);
}

// Compare "moving" part of two InputSections, namely relocation targets.
template <class ELFT>
static bool equalsVariable(const InputSection<ELFT> *A,
                           const InputSection<ELFT> *B) {
  if (A->AreRelocsRela)
    return variableEq(A, A->relas(), B, B->relas());
  return variableEq(A, A->rels(), B, B->rels());
}

// The main function of ICF.
template <class ELFT> void ICF<ELFT>::run() {
  // Initially, we use hash values as section group IDs. Therefore,
  // if two sections have the same ID, they are likely (but not
  // guaranteed) to have the same static contents in terms of ICF.
  std::vector<InputSection<ELFT> *> Sections = getSections<ELFT>();
  for (InputSection<ELFT> *S : Sections)
    // Set MSB on to avoid collisions with serial group IDs
    S->GroupId = getHash(S) | (uint64_t(1) << 63);

  // From now on, sections in V are ordered so that sections in
  // the same group are consecutive in the vector.
  std::stable_sort(Sections.begin(), Sections.end(),
                   [](InputSection<ELFT> *A, InputSection<ELFT> *B) {
                     if (A->GroupId != B->GroupId)
                       return A->GroupId < B->GroupId;
                     // Within a group, put the highest alignment
                     // requirement first, so that's the one we'll keep.
                     return B->Alignment < A->Alignment;
                   });

  // Compare static contents and assign unique IDs for each static content.
  forEachGroup(Sections, [&](MutableArrayRef<InputSection<ELFT> *> V) {
    segregate(V, equalsConstant<ELFT>);
  });

  // Split groups by comparing relocations until we get a convergence.
  int Cnt = 1;
  for (;;) {
    ++Cnt;
    uint64_t Id = NextId;
    forEachGroup(Sections, [&](MutableArrayRef<InputSection<ELFT> *> V) {
      segregate(V, equalsVariable<ELFT>);
    });
    if (Id == NextId)
      break;
  }
  log("ICF needed " + Twine(Cnt) + " iterations.");

  // Merge sections in the same group.
  forEachGroup(Sections, [](MutableArrayRef<InputSection<ELFT> *> V) {
    log("selected " + V[0]->Name);
    for (InputSection<ELFT> *S : V.slice(1)) {
      log("  removed " + S->Name);
      V[0]->replace(S);
    }
  });
}

// ICF entry point function.
template <class ELFT> void elf::doIcf() { ICF<ELFT>().run(); }

template void elf::doIcf<ELF32LE>();
template void elf::doIcf<ELF32BE>();
template void elf::doIcf<ELF64LE>();
template void elf::doIcf<ELF64BE>();
