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
struct Range {
  size_t Begin;
  size_t End;
};

template <class ELFT> class ICF {
public:
  void run();

private:
  void segregate(Range *R, bool Constant);

  template <class RelTy>
  bool constantEq(ArrayRef<RelTy> RelsA, ArrayRef<RelTy> RelsB);

  template <class RelTy>
  bool variableEq(const InputSection<ELFT> *A, ArrayRef<RelTy> RelsA,
                  const InputSection<ELFT> *B, ArrayRef<RelTy> RelsB);

  bool equalsConstant(const InputSection<ELFT> *A, const InputSection<ELFT> *B);
  bool equalsVariable(const InputSection<ELFT> *A, const InputSection<ELFT> *B);

  std::vector<InputSection<ELFT> *> Sections;
  std::vector<Range> Ranges;

  // The main loop is repeated until we get a convergence.
  bool Repeat = false; // If Repeat is true, we need to repeat.
  int Cnt = 0;         // Counter for the main loop.
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

// Before calling this function, all sections in range R must have the
// same group ID.
template <class ELFT> void ICF<ELFT>::segregate(Range *R, bool Constant) {
  // This loop rearranges sections in range R so that all sections
  // that are equal in terms of equals{Constant,Variable} are contiguous
  // in Sections vector.
  //
  // The algorithm is quadratic in the worst case, but that is not an
  // issue in practice because the number of the distinct sections in
  // [R.Begin, R.End] is usually very small.
  while (R->End - R->Begin > 1) {
    // Divide range R into two. Let Mid be the start index of the
    // second group.
    auto Bound = std::stable_partition(
        Sections.begin() + R->Begin + 1, Sections.begin() + R->End,
        [&](InputSection<ELFT> *S) {
          if (Constant)
            return equalsConstant(Sections[R->Begin], S);
          return equalsVariable(Sections[R->Begin], S);
        });
    size_t Mid = Bound - Sections.begin();

    if (Mid == R->End)
      return;

    // Now we split [R.Begin, R.End) into [R.Begin, Mid) and [Mid, R.End).
    if (Mid - R->Begin > 1)
      Ranges.push_back({R->Begin, Mid});
    R->Begin = Mid;

    // Update GroupIds for the new group members. We use the index of
    // the group first member as a group ID because that is unique.
    for (size_t I = Mid; I < R->End; ++I)
      Sections[I]->GroupId = Mid;

    // Since we have split a group, we need to repeat the main loop
    // later to obtain a convergence. Remember that.
    Repeat = true;
  }
}

// Compare two lists of relocations.
template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::constantEq(ArrayRef<RelTy> RelsA, ArrayRef<RelTy> RelsB) {
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
bool ICF<ELFT>::equalsConstant(const InputSection<ELFT> *A,
                               const InputSection<ELFT> *B) {
  if (A->NumRelocations != B->NumRelocations || A->Flags != B->Flags ||
      A->getSize() != B->getSize() || A->Data != B->Data)
    return false;

  if (A->AreRelocsRela)
    return constantEq(A->relas(), B->relas());
  return constantEq(A->rels(), B->rels());
}

// Compare two lists of relocations. Returns true if all pairs of
// relocations point to the same section in terms of ICF.
template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::variableEq(const InputSection<ELFT> *A, ArrayRef<RelTy> RelsA,
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
    if (!X || !Y)
      return false;
    return X->GroupId != 0 && X->GroupId == Y->GroupId;
  };

  return std::equal(RelsA.begin(), RelsA.end(), RelsB.begin(), Eq);
}

// Compare "moving" part of two InputSections, namely relocation targets.
template <class ELFT>
bool ICF<ELFT>::equalsVariable(const InputSection<ELFT> *A,
                               const InputSection<ELFT> *B) {
  if (A->AreRelocsRela)
    return variableEq(A, A->relas(), B, B->relas());
  return variableEq(A, A->rels(), B, B->rels());
}

// The main function of ICF.
template <class ELFT> void ICF<ELFT>::run() {
  // Collect sections to merge.
  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections)
    if (auto *S = dyn_cast<InputSection<ELFT>>(Sec))
      if (isEligible(S))
        Sections.push_back(S);

  // Initially, we use hash values as section group IDs. Therefore,
  // if two sections have the same ID, they are likely (but not
  // guaranteed) to have the same static contents in terms of ICF.
  for (InputSection<ELFT> *S : Sections)
    // Set MSB to 1 to avoid collisions with non-hash IDs.
    S->GroupId = getHash(S) | (uint64_t(1) << 63);

  // From now on, sections in Sections are ordered so that sections in
  // the same group are consecutive in the vector.
  std::stable_sort(Sections.begin(), Sections.end(),
                   [](InputSection<ELFT> *A, InputSection<ELFT> *B) {
                     if (A->GroupId != B->GroupId)
                       return A->GroupId < B->GroupId;
                     // Within a group, put the highest alignment
                     // requirement first, so that's the one we'll keep.
                     return B->Alignment < A->Alignment;
                   });

  // Split sections into groups by ID. And then we are going to
  // split groups into more and more smaller groups.
  // Note that we do not add single element groups because they
  // are already the smallest.
  Ranges.reserve(Sections.size());
  for (size_t I = 0, E = Sections.size(); I < E - 1;) {
    // Let J be the first index whose element has a different ID.
    size_t J = I + 1;
    while (J < E && Sections[I]->GroupId == Sections[J]->GroupId)
      ++J;
    if (J - I > 1)
      Ranges.push_back({I, J});
    I = J;
  }

  // Compare static contents and assign unique IDs for each static content.
  std::for_each(Ranges.begin(), Ranges.end(),
                [&](Range &R) { segregate(&R, true); });
  ++Cnt;

  // Split groups by comparing relocations until convergence is obtained.
  do {
    Repeat = false;
    std::for_each(Ranges.begin(), Ranges.end(),
                  [&](Range &R) { segregate(&R, false); });
    ++Cnt;
  } while (Repeat);

  log("ICF needed " + Twine(Cnt) + " iterations");

  // Merge sections in the same group.
  for (Range R : Ranges) {
    if (R.End - R.Begin == 1)
      continue;

    log("selected " + Sections[R.Begin]->Name);
    for (size_t I = R.Begin + 1; I < R.End; ++I) {
      log("  removed " + Sections[I]->Name);
      Sections[R.Begin]->replace(Sections[I]);
    }
  }
}

// ICF entry point function.
template <class ELFT> void elf::doIcf() { ICF<ELFT>().run(); }

template void elf::doIcf<ELF32LE>();
template void elf::doIcf<ELF32BE>();
template void elf::doIcf<ELF64LE>();
template void elf::doIcf<ELF64BE>();
