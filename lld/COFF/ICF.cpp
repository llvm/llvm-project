//===- ICF.cpp ------------------------------------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Identical COMDAT Folding is a feature to merge COMDAT sections not by
// name (which is regular COMDAT handling) but by contents. If two COMDAT
// sections have the same data, relocations, attributes, etc., then the two
// are considered identical and merged by the linker. This optimization
// makes outputs smaller.
//
// ICF is theoretically a problem of reducing graphs by merging as many
// identical subgraphs as possible, if we consider sections as vertices and
// relocations as edges. This may be a bit more complicated problem than you
// might think. The order of processing sections matters since merging two
// sections can make other sections, whose relocations now point to the same
// section, mergeable. Graphs may contain cycles, which is common in COFF.
// We need a sophisticated algorithm to do this properly and efficiently.
//
// What we do in this file is this. We split sections into groups. Sections
// in the same group are considered identical.
//
// First, all sections are grouped by their "constant" values. Constant
// values are values that are never changed by ICF, such as section contents,
// section name, number of relocations, type and offset of each relocation,
// etc. Because we do not care about some relocation targets in this step,
// two sections in the same group may not be identical, but at least two
// sections in different groups can never be identical.
//
// Then, we try to split each group by relocation targets. Relocations are
// considered identical if and only if the relocation targets are in the
// same group. Splitting a group may make more groups to be splittable,
// because two relocations that were previously considered identical might
// now point to different groups. We repeat this step until the convergence
// is obtained.
//
// This algorithm is so-called "optimistic" algorithm described in
// http://research.google.com/pubs/pub36912.html.
//
//===----------------------------------------------------------------------===//

#include "Chunks.h"
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

typedef std::vector<SectionChunk *>::iterator ChunkIterator;
typedef bool (*Comparator)(const SectionChunk *, const SectionChunk *);

class ICF {
public:
  void run(const std::vector<Chunk *> &V);

private:
  static uint64_t getHash(SectionChunk *C);
  static bool equalsConstant(const SectionChunk *A, const SectionChunk *B);
  static bool equalsVariable(const SectionChunk *A, const SectionChunk *B);
  bool forEachGroup(std::vector<SectionChunk *> &Chunks, Comparator Eq);
  bool partition(ChunkIterator Begin, ChunkIterator End, Comparator Eq);

  std::atomic<uint64_t> NextID = { 1 };
};

// Entry point to ICF.
void doICF(const std::vector<Chunk *> &Chunks) {
  ICF().run(Chunks);
}

uint64_t ICF::getHash(SectionChunk *C) {
  return hash_combine(C->getPermissions(),
                      hash_value(C->SectionName),
                      C->NumRelocs,
                      C->getAlign(),
                      uint32_t(C->Header->SizeOfRawData),
                      C->Checksum);
}

bool ICF::equalsConstant(const SectionChunk *A, const SectionChunk *B) {
  if (A->AssocChildren.size() != B->AssocChildren.size() ||
      A->NumRelocs != B->NumRelocs) {
    return false;
  }

  // Compare associative sections.
  for (size_t I = 0, E = A->AssocChildren.size(); I != E; ++I)
    if (A->AssocChildren[I]->GroupID != B->AssocChildren[I]->GroupID)
      return false;

  // Compare relocations.
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    if (R1.Type != R2.Type ||
        R1.VirtualAddress != R2.VirtualAddress) {
      return false;
    }
    SymbolBody *B1 = A->File->getSymbolBody(R1.SymbolTableIndex)->repl();
    SymbolBody *B2 = B->File->getSymbolBody(R2.SymbolTableIndex)->repl();
    if (B1 == B2)
      return true;
    if (auto *D1 = dyn_cast<DefinedRegular>(B1))
      if (auto *D2 = dyn_cast<DefinedRegular>(B2))
        return D1->getValue() == D2->getValue() &&
               D1->getChunk()->GroupID == D2->getChunk()->GroupID;
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

bool ICF::equalsVariable(const SectionChunk *A, const SectionChunk *B) {
  // Compare associative sections.
  for (size_t I = 0, E = A->AssocChildren.size(); I != E; ++I)
    if (A->AssocChildren[I]->GroupID != B->AssocChildren[I]->GroupID)
      return false;

  // Compare relocations.
  auto Eq = [&](const coff_relocation &R1, const coff_relocation &R2) {
    SymbolBody *B1 = A->File->getSymbolBody(R1.SymbolTableIndex)->repl();
    SymbolBody *B2 = B->File->getSymbolBody(R2.SymbolTableIndex)->repl();
    if (B1 == B2)
      return true;
    if (auto *D1 = dyn_cast<DefinedRegular>(B1))
      if (auto *D2 = dyn_cast<DefinedRegular>(B2))
        return D1->getChunk()->GroupID == D2->getChunk()->GroupID;
    return false;
  };
  return std::equal(A->Relocs.begin(), A->Relocs.end(), B->Relocs.begin(), Eq);
}

bool ICF::partition(ChunkIterator Begin, ChunkIterator End, Comparator Eq) {
  bool R = false;
  for (auto It = Begin;;) {
    SectionChunk *Head = *It;
    auto Bound = std::partition(It + 1, End, [&](SectionChunk *SC) {
      return Eq(Head, SC);
    });
    if (Bound == End)
      return R;
    uint64_t ID = NextID++;
    std::for_each(It, Bound, [&](SectionChunk *SC) { SC->GroupID = ID; });
    It = Bound;
    R = true;
  }
}

bool ICF::forEachGroup(std::vector<SectionChunk *> &Chunks, Comparator Eq) {
  bool R = false;
  for (auto It = Chunks.begin(), End = Chunks.end(); It != End;) {
    SectionChunk *Head = *It;
    auto Bound = std::find_if(It + 1, End, [&](SectionChunk *SC) {
      return SC->GroupID != Head->GroupID;
    });
    if (partition(It, Bound, Eq))
      R = true;
    It = Bound;
  }
  return R;
}

// Merge identical COMDAT sections.
// Two sections are considered the same if their section headers,
// contents and relocations are all the same.
void ICF::run(const std::vector<Chunk *> &Vec) {
  // Collect only mergeable sections and group by hash value.
  parallel_for_each(Vec.begin(), Vec.end(), [&](Chunk *C) {
    if (auto *SC = dyn_cast<SectionChunk>(C)) {
      bool Global = SC->Sym && SC->Sym->isExternal();
      bool Writable = SC->getPermissions() & llvm::COFF::IMAGE_SCN_MEM_WRITE;
      if (SC->isCOMDAT() && SC->isLive() && Global && !Writable)
        SC->GroupID = getHash(SC) | (uint64_t(1) << 63);
    }
  });
  std::vector<SectionChunk *> Chunks;
  for (Chunk *C : Vec) {
    if (auto *SC = dyn_cast<SectionChunk>(C)) {
      if (SC->GroupID) {
        Chunks.push_back(SC);
      } else {
        SC->GroupID = NextID++;
      }
    }
  }

  // From now on, sections in Chunks are ordered so that sections in
  // the same group are consecutive in the vector.
  std::sort(Chunks.begin(), Chunks.end(),
            [](SectionChunk *A, SectionChunk *B) {
              return A->GroupID < B->GroupID;
            });

  // Split groups until we get a convergence.
  int Cnt = 1;
  forEachGroup(Chunks, equalsConstant);

  for (;;) {
    if (!forEachGroup(Chunks, equalsVariable))
      break;
    ++Cnt;
  }
  if (Config->Verbose)
    llvm::outs() << "\nICF needed " << Cnt << " iterations.\n";

  // Merge sections in the same group.
  for (auto It = Chunks.begin(), End = Chunks.end(); It != End;) {
    SectionChunk *Head = *It++;
    auto Bound = std::find_if(It, End, [&](SectionChunk *SC) {
      return Head->GroupID != SC->GroupID;
    });
    if (It == Bound)
      continue;
    if (Config->Verbose)
      llvm::outs() << "Selected " << Head->getDebugName() << "\n";
    while (It != Bound) {
      SectionChunk *SC = *It++;
      if (Config->Verbose)
        llvm::outs() << "  Removed " << SC->getDebugName() << "\n";
      Head->replace(SC);
    }
  }
}

} // namespace coff
} // namespace lld
