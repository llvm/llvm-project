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
// On Windows, ICF is enabled by default.
//
// See ELF/ICF.cpp for the details about the algortihm.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
#include "Error.h"
#include "Symbols.h"
#include "lld/Core/Parallel.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <atomic>
#include <vector>

using namespace llvm;

namespace lld {
namespace coff {

class ICF {
public:
  void run(const std::vector<Chunk *> &V);

private:
  void segregate(size_t Begin, size_t End, bool Constant);

  bool equalsConstant(const SectionChunk *A, const SectionChunk *B);
  bool equalsVariable(const SectionChunk *A, const SectionChunk *B);

  uint32_t getHash(SectionChunk *C);
  bool isEligible(SectionChunk *C);

  size_t findBoundary(size_t Begin, size_t End);

  void forEachColorRange(size_t Begin, size_t End,
                         std::function<void(size_t, size_t)> Fn);

  void forEachColor(std::function<void(size_t, size_t)> Fn);

  std::vector<SectionChunk *> Chunks;
  int Cnt = 0;
  std::atomic<uint32_t> NextId = {1};
  std::atomic<bool> Repeat = {false};
};

// Returns a hash value for S.
uint32_t ICF::getHash(SectionChunk *C) {
  return hash_combine(C->getPermissions(),
                      hash_value(C->SectionName),
                      C->NumRelocs,
                      C->getAlign(),
                      uint32_t(C->Header->SizeOfRawData),
                      C->Checksum);
}

// Returns true if section S is subject of ICF.
bool ICF::isEligible(SectionChunk *C) {
  bool Global = C->Sym && C->Sym->isExternal();
  bool Writable = C->getPermissions() & llvm::COFF::IMAGE_SCN_MEM_WRITE;
  return C->isCOMDAT() && C->isLive() && Global && !Writable;
}

// Split a range into smaller ranges by recoloring sections
void ICF::segregate(size_t Begin, size_t End, bool Constant) {
  while (Begin < End) {
    // Divide [Begin, End) into two. Let Mid be the start index of the
    // second group.
    auto Bound = std::stable_partition(
        Chunks.begin() + Begin + 1, Chunks.begin() + End, [&](SectionChunk *S) {
          if (Constant)
            return equalsConstant(Chunks[Begin], S);
          return equalsVariable(Chunks[Begin], S);
        });
    size_t Mid = Bound - Chunks.begin();

    // Split [Begin, End) into [Begin, Mid) and [Mid, End).
    uint32_t Id = NextId++;
    for (size_t I = Begin; I < Mid; ++I)
      Chunks[I]->Color[(Cnt + 1) % 2] = Id;

    // If we created a group, we need to iterate the main loop again.
    if (Mid != End)
      Repeat = true;

    Begin = Mid;
  }
}

// Compare "non-moving" part of two sections, namely everything
// except relocation targets.
bool ICF::equalsConstant(const SectionChunk *A, const SectionChunk *B) {
  if (A->NumRelocs != B->NumRelocs)
    return false;

  // Compare relocations.
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    if (R1.Type != R2.Type ||
        R1.VirtualAddress != R2.VirtualAddress) {
      return false;
    }
    SymbolBody *B1 = A->File->getSymbolBody(R1.SymbolTableIndex);
    SymbolBody *B2 = B->File->getSymbolBody(R2.SymbolTableIndex);
    if (B1 == B2)
      return true;
    if (auto *D1 = dyn_cast<DefinedRegular>(B1))
      if (auto *D2 = dyn_cast<DefinedRegular>(B2))
        return D1->getValue() == D2->getValue() &&
               D1->getChunk()->Color[Cnt % 2] == D2->getChunk()->Color[Cnt % 2];
    return false;
  };
  if (!std::equal(A->Relocs.begin(), A->Relocs.end(), B->Relocs.begin(), Eq))
    return false;

  // Compare section attributes and contents.
  return A->getPermissions() == B->getPermissions() &&
         A->SectionName == B->SectionName &&
         A->getAlign() == B->getAlign() &&
         A->Header->SizeOfRawData == B->Header->SizeOfRawData &&
         A->Checksum == B->Checksum &&
         A->getContents() == B->getContents();
}

// Compare "moving" part of two sections, namely relocation targets.
bool ICF::equalsVariable(const SectionChunk *A, const SectionChunk *B) {
  // Compare relocations.
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    SymbolBody *B1 = A->File->getSymbolBody(R1.SymbolTableIndex);
    SymbolBody *B2 = B->File->getSymbolBody(R2.SymbolTableIndex);
    if (B1 == B2)
      return true;
    if (auto *D1 = dyn_cast<DefinedRegular>(B1))
      if (auto *D2 = dyn_cast<DefinedRegular>(B2))
        return D1->getChunk()->Color[Cnt % 2] == D2->getChunk()->Color[Cnt % 2];
    return false;
  };
  return std::equal(A->Relocs.begin(), A->Relocs.end(), B->Relocs.begin(), Eq);
}

size_t ICF::findBoundary(size_t Begin, size_t End) {
  for (size_t I = Begin + 1; I < End; ++I)
    if (Chunks[Begin]->Color[Cnt % 2] != Chunks[I]->Color[Cnt % 2])
      return I;
  return End;
}

void ICF::forEachColorRange(size_t Begin, size_t End,
                            std::function<void(size_t, size_t)> Fn) {
  if (Begin > 0)
    Begin = findBoundary(Begin - 1, End);

  while (Begin < End) {
    size_t Mid = findBoundary(Begin, Chunks.size());
    Fn(Begin, Mid);
    Begin = Mid;
  }
}

// Call Fn on each color group.
void ICF::forEachColor(std::function<void(size_t, size_t)> Fn) {
  // If the number of sections are too small to use threading,
  // call Fn sequentially.
  if (Chunks.size() < 1024) {
    forEachColorRange(0, Chunks.size(), Fn);
    return;
  }

  // Split sections into 256 shards and call Fn in parallel.
  size_t NumShards = 256;
  size_t Step = Chunks.size() / NumShards;
  parallel_for(size_t(0), NumShards, [&](size_t I) {
    forEachColorRange(I * Step, (I + 1) * Step, Fn);
  });
  forEachColorRange(Step * NumShards, Chunks.size(), Fn);
}

// Merge identical COMDAT sections.
// Two sections are considered the same if their section headers,
// contents and relocations are all the same.
void ICF::run(const std::vector<Chunk *> &Vec) {
  // Collect only mergeable sections and group by hash value.
  for (Chunk *C : Vec) {
    auto *SC = dyn_cast<SectionChunk>(C);
    if (!SC)
      continue;

    if (isEligible(SC)) {
      // Set MSB to 1 to avoid collisions with non-hash colors.
      SC->Color[0] = getHash(SC) | (1 << 31);
      Chunks.push_back(SC);
    } else {
      SC->Color[0] = NextId++;
    }
  }

  if (Chunks.empty())
    return;

  // From now on, sections in Chunks are ordered so that sections in
  // the same group are consecutive in the vector.
  std::stable_sort(Chunks.begin(), Chunks.end(),
                   [](SectionChunk *A, SectionChunk *B) {
                     return A->Color[0] < B->Color[0];
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

  if (Config->Verbose)
    outs() << "\nICF needed " << Cnt << " iterations\n";

  // Merge sections in the same colors.
  forEachColor([&](size_t Begin, size_t End) {
    if (End - Begin == 1)
      return;

    if (Config->Verbose)
      outs() << "Selected " << Chunks[Begin]->getDebugName() << "\n";
    for (size_t I = Begin + 1; I < End; ++I) {
      if (Config->Verbose)
        outs() << "  Removed " << Chunks[I]->getDebugName() << "\n";
      Chunks[Begin]->replace(Chunks[I]);
    }
  });
}

// Entry point to ICF.
void doICF(const std::vector<Chunk *> &Chunks) { ICF().run(Chunks); }

} // namespace coff
} // namespace lld
