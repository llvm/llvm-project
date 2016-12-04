//===- ICF.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// ICF is short for Identical Code Folding. That is a size optimization to
// identify and merge two or more read-only sections (typically functions)
// that happened to have the same contents. It usually reduces output size
// by a few percent.
//
// In ICF, two sections are considered identical if they have the same
// section flags, section data, and relocations. Relocations are tricky,
// because two relocations are considered the same if they have the same
// relocation types, values, and if they point to the same sections *in
// terms of ICF*.
//
// Here is an example. If foo and bar defined below are compiled to the
// same machine instructions, ICF can and should merge the two, although
// their relocations point to each other.
//
//   void foo() { bar(); }
//   void bar() { foo(); }
//
// If you merge the two, their relocations point to the same section and
// thus you know they are mergeable, but how do we know they are mergeable
// in the first place? This is not an easy problem to solve.
//
// What we are doing in LLD is some sort of coloring algorithm.
//
// We color non-identical sections in different colors repeatedly.
// Sections in the same color when the algorithm terminates are considered
// identical. Here are the details:
//
// 1. First, we color all sections using their hash values of section
//    types, section contents, and numbers of relocations. At this moment,
//    relocation targets are not taken into account. We just color
//    sections that apparently differ in different colors.
//
// 2. Next, for each color C, we visit sections in color C to compare
//    relocation target colors.  We recolor sections A and B in different
//    colors if A's and B's relocations are different in terms of target
//    colors.
//
// 3. If we recolor some section in step 2, relocations that were
//    previously pointing to the same color targets may now be pointing to
//    different colors. Therefore, repeat 2 until a convergence is
//    obtained.
//
// 4. For each color C, pick an arbitrary section in color C, and merges
//    other sections in color C with it.
//
// For small programs, this algorithm needs 3-5 iterations. For large
// programs such as Chromium, it takes more than 20 iterations.
//
// We parallelize each step so that multiple threads can work on different
// colors concurrently. That gave us a large performance boost when
// applying ICF on large programs. For example, MSVC link.exe or GNU gold
// takes 10-20 seconds to apply ICF on Chromium, whose output size is
// about 1.5 GB, but LLD can finish it in less than 2 seconds on a 2.8 GHz
// 40 core machine. Even without threading, LLD's ICF is still faster than
// MSVC or gold though.
//
//===----------------------------------------------------------------------===//

#include "ICF.h"
#include "Config.h"
#include "SymbolTable.h"
#include "Threads.h"

#include "llvm/ADT/Hashing.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/ELF.h"
#include <algorithm>
#include <atomic>

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
  void segregate(size_t Begin, size_t End, bool Constant);

  template <class RelTy>
  bool constantEq(ArrayRef<RelTy> RelsA, ArrayRef<RelTy> RelsB);

  template <class RelTy>
  bool variableEq(const InputSection<ELFT> *A, ArrayRef<RelTy> RelsA,
                  const InputSection<ELFT> *B, ArrayRef<RelTy> RelsB);

  bool equalsConstant(const InputSection<ELFT> *A, const InputSection<ELFT> *B);
  bool equalsVariable(const InputSection<ELFT> *A, const InputSection<ELFT> *B);

  size_t findBoundary(size_t Begin, size_t End);

  void forEachColorRange(size_t Begin, size_t End,
                         std::function<void(size_t, size_t)> Fn);

  void forEachColor(std::function<void(size_t, size_t)> Fn);

  std::vector<InputSection<ELFT> *> Sections;
  int Cnt = 0;
  std::atomic<bool> Repeat = {false};
};
}

// Returns a hash value for S. Note that the information about
// relocation targets is not included in the hash value.
template <class ELFT> static uint32_t getHash(InputSection<ELFT> *S) {
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

// Split a range into smaller ranges by recoloring sections
// in a given range.
template <class ELFT>
void ICF<ELFT>::segregate(size_t Begin, size_t End, bool Constant) {
  // This loop rearranges sections in [Begin, End) so that all sections
  // that are equal in terms of equals{Constant,Variable} are contiguous
  // in [Begin, End).
  //
  // The algorithm is quadratic in the worst case, but that is not an
  // issue in practice because the number of the distinct sections in
  // each range is usually very small.

  while (Begin < End) {
    // Divide [Begin, End) into two. Let Mid be the start index of the
    // second group.
    auto Bound = std::stable_partition(
        Sections.begin() + Begin + 1, Sections.begin() + End,
        [&](InputSection<ELFT> *S) {
          if (Constant)
            return equalsConstant(Sections[Begin], S);
          return equalsVariable(Sections[Begin], S);
        });
    size_t Mid = Bound - Sections.begin();

    // Now we split [Begin, End) into [Begin, Mid) and [Mid, End) by
    // updating the section colors in [Begin, End). We use Mid as a
    // color ID because every group ends with a unique index.
    //
    // Note on Color[0] and Color[1]: we have two storages for colors.
    // At the beginning of each iteration of the main loop, both have
    // the same color. Color[0] contains the current color, and Color[1]
    // contains the next color which will be used on the next iteration.
    //
    // Recall that other threads may be working on other ranges. They
    // may be reading colors that we are about to update. We cannot
    // update colors in place because it breaks the invariance that
    // all sections in the same group must have the same color. In
    // other words, the following for loop is not an atomic operation,
    // and that is observable from other threads.
    //
    // By writing new colors to write-only places, we can keep the invariance.
    for (size_t I = Begin; I < Mid; ++I)
      Sections[I]->Color[(Cnt + 1) % 2] = Mid;

    // If we created a group, we need to iterate the main loop again.
    if (Mid != End)
      Repeat = true;

    Begin = Mid;
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
    // The two sections must be identical.
    SymbolBody &SA = A->getFile()->getRelocTargetSym(RA);
    SymbolBody &SB = B->getFile()->getRelocTargetSym(RB);
    if (&SA == &SB)
      return true;

    // Or, the two sections must have the same color.
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

    // Ineligible sections have the special color 0.
    // They can never be the same in terms of section colors.
    if (X->Color[Cnt % 2] == 0)
      return false;

    return X->Color[Cnt % 2] == Y->Color[Cnt % 2];
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

template <class ELFT> size_t ICF<ELFT>::findBoundary(size_t Begin, size_t End) {
  for (size_t I = Begin + 1; I < End; ++I)
    if (Sections[Begin]->Color[Cnt % 2] != Sections[I]->Color[Cnt % 2])
      return I;
  return End;
}

// Sections in the same color are contiguous in Sections vector.
// Therefore, Sections vector can be considered as contiguous groups
// of sections, grouped by colors.
//
// This function calls Fn on every group that starts within [Begin, End).
// Note that a group must starts in that range but doesn't necessarily
// have to end before End.
template <class ELFT>
void ICF<ELFT>::forEachColorRange(size_t Begin, size_t End,
                                  std::function<void(size_t, size_t)> Fn) {
  if (Begin > 0)
    Begin = findBoundary(Begin - 1, End);

  while (Begin < End) {
    size_t Mid = findBoundary(Begin, Sections.size());
    Fn(Begin, Mid);
    Begin = Mid;
  }
}

// Call Fn on each color group.
template <class ELFT>
void ICF<ELFT>::forEachColor(std::function<void(size_t, size_t)> Fn) {
  // If threading is disabled or the number of sections are
  // too small to use threading, call Fn sequentially.
  if (!Config->Threads || Sections.size() < 1024) {
    forEachColorRange(0, Sections.size(), Fn);
    return;
  }

  // Split sections into 256 shards and call Fn in parallel.
  size_t NumShards = 256;
  size_t Step = Sections.size() / NumShards;
  forLoop(0, NumShards,
          [&](size_t I) { forEachColorRange(I * Step, (I + 1) * Step, Fn); });
  forEachColorRange(Step * NumShards, Sections.size(), Fn);
}

// The main function of ICF.
template <class ELFT> void ICF<ELFT>::run() {
  // Collect sections to merge.
  for (InputSectionBase<ELFT> *Sec : Symtab<ELFT>::X->Sections)
    if (auto *S = dyn_cast<InputSection<ELFT>>(Sec))
      if (isEligible(S))
        Sections.push_back(S);

  // Initially, we use hash values to color sections. Therefore, if
  // two sections have the same color, they are likely (but not
  // guaranteed) to have the same static contents in terms of ICF.
  for (InputSection<ELFT> *S : Sections)
    // Set MSB to 1 to avoid collisions with non-hash colors.
    S->Color[0] = getHash(S) | (1 << 31);

  // From now on, sections in Sections are ordered so that sections in
  // the same color are consecutive in the vector.
  std::stable_sort(Sections.begin(), Sections.end(),
                   [](InputSection<ELFT> *A, InputSection<ELFT> *B) {
                     if (A->Color[0] != B->Color[0])
                       return A->Color[0] < B->Color[0];
                     // Within a group, put the highest alignment
                     // requirement first, so that's the one we'll keep.
                     return B->Alignment < A->Alignment;
                   });

  // Compare static contents and assign unique IDs for each static content.
  forEachColor([&](size_t Begin, size_t End) { segregate(Begin, End, true); });
  ++Cnt;

  // Split groups by comparing relocations until convergence is obtained.
  do {
    Repeat = false;
    forEachColor(
        [&](size_t Begin, size_t End) { segregate(Begin, End, false); });
    ++Cnt;
  } while (Repeat);

  log("ICF needed " + Twine(Cnt) + " iterations");

  // Merge sections in the same colors.
  forEachColor([&](size_t Begin, size_t End) {
    if (End - Begin == 1)
      return;

    log("selected " + Sections[Begin]->Name);
    for (size_t I = Begin + 1; I < End; ++I) {
      log("  removed " + Sections[I]->Name);
      Sections[Begin]->replace(Sections[I]);
    }
  });
}

// ICF entry point function.
template <class ELFT> void elf::doIcf() { ICF<ELFT>().run(); }

template void elf::doIcf<ELF32LE>();
template void elf::doIcf<ELF32BE>();
template void elf::doIcf<ELF64LE>();
template void elf::doIcf<ELF64BE>();
