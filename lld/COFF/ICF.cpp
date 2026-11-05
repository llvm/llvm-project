//===- ICF.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// See ELF/ICF.cpp for the details about the algorithm.
//
//===----------------------------------------------------------------------===//

#include "ICF.h"
#include "COFFLinkerContext.h"
#include "Chunks.h"
#include "Symbols.h"
#include "lld/Common/Timer.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <atomic>
#include <vector>

using namespace llvm;

namespace lld::coff {
// Propagation mixes a self-referencing section's hash with itself, so mix(h, h)
// must not be a constant.
static uint64_t mix(uint64_t h, uint64_t v) {
  h = (h + v) * 0x9e3779b97f4a7c15;
  return h ^ (h >> 32);
}

// Ignore associated metadata sections that don't participate in ICF, such as
// debug info and CFGuard metadata.
static bool considerForICF(const SectionChunk &assoc) {
  StringRef name = assoc.getSectionName();
  return !(name.starts_with(".debug") || name == ".gfids$y" ||
           name == ".giats$y" || name == ".gljmp$y");
}

class ICF {
public:
  ICF(COFFLinkerContext &c) : ctx(c) {}
  void run();

private:
  template <bool constant> void segregate(size_t begin, size_t end);

  bool assocEquals(const SectionChunk *a, const SectionChunk *b);

  template <bool constant>
  bool sectionsEqual(const SectionChunk *a, const SectionChunk *b);

  bool isEligible(SectionChunk *c);

  size_t findBoundary(size_t begin, size_t end);

  void forEachClassRange(size_t begin, size_t end,
                         std::function<void(size_t, size_t)> fn);

  void forEachClass(std::function<void(size_t, size_t)> fn);

  std::vector<SectionChunk *> chunks;
  int cnt = 0;
  std::atomic<bool> repeat = {false};

  COFFLinkerContext &ctx;
};

// Returns true if section S is subject of ICF.
//
// Microsoft's documentation
// (https://msdn.microsoft.com/en-us/library/bxwfs976.aspx; visited April
// 2017) says that /opt:icf folds both functions and read-only data.
// Despite that, the MSVC linker folds only functions. We found
// a few instances of programs that are not safe for data merging.
// Therefore, we merge only functions just like the MSVC tool. However, we also
// merge read-only sections in a couple of cases where the address of the
// section is insignificant to the user program and the behaviour matches that
// of the Visual C++ linker.
bool ICF::isEligible(SectionChunk *c) {
  // Non-comdat chunks, dead chunks, and writable chunks are not eligible.
  bool writable =
      c->getOutputCharacteristics() & llvm::COFF::IMAGE_SCN_MEM_WRITE;
  if (!c->isCOMDAT() || !c->live || writable)
    return false;

  // Under regular (not safe) ICF, all code sections are eligible.
  if ((ctx.config.doICF == ICFLevel::All) &&
      c->getOutputCharacteristics() & llvm::COFF::IMAGE_SCN_MEM_EXECUTE)
    return true;

  // .pdata and .xdata unwind info sections are eligible.
  StringRef outSecName = c->getSectionName().split('$').first;
  if (outSecName == ".pdata" || outSecName == ".xdata")
    return true;

  // So are vtables.
  const char *itaniumVtablePrefix =
      ctx.config.machine == I386 ? "__ZTV" : "_ZTV";
  if (c->sym && (c->sym->getName().starts_with("??_7") ||
                 c->sym->getName().starts_with(itaniumVtablePrefix)))
    return true;

  // Anything else not in an address-significance table is eligible.
  return !c->keepUnique;
}

// Split an equivalence class into smaller classes.
template <bool constant> void ICF::segregate(size_t begin, size_t end) {
  while (begin < end) {
    // Divide [Begin, End) into two. Let Mid be the start index of the
    // second group.
    auto bound = std::stable_partition(
        chunks.begin() + begin + 1, chunks.begin() + end, [&](SectionChunk *s) {
          return sectionsEqual<constant>(chunks[begin], s);
        });
    size_t mid = bound - chunks.begin();

    // Split [Begin, End) into [Begin, Mid) and [Mid, End). We use Mid as an
    // equivalence class ID because every group ends with a unique index.
    for (size_t i = begin; i < mid; ++i)
      chunks[i]->eqClass[(cnt + 1) % 2] = mid;

    // If we created a group, we need to iterate the main loop again.
    if (mid != end)
      repeat = true;

    begin = mid;
  }
}

// Returns true if two sections' associative children, i.e. exception handling
// metadata such as .pdata and .xdata, are equal.
bool ICF::assocEquals(const SectionChunk *a, const SectionChunk *b) {
  auto ra = make_filter_range(a->children(), considerForICF);
  auto rb = make_filter_range(b->children(), considerForICF);
  return std::equal(ra.begin(), ra.end(), rb.begin(), rb.end(),
                    [&](const SectionChunk &ia, const SectionChunk &ib) {
                      return ia.eqClass[cnt % 2] == ib.eqClass[cnt % 2];
                    });
}

// Compare the "non-moving" or "moving" parts of two sections.
template <bool constant>
bool ICF::sectionsEqual(const SectionChunk *a, const SectionChunk *b) {
  auto eqSym = [&](Symbol *b1, Symbol *b2) {
    if (b1 == b2)
      return true;
    auto *d1 = dyn_cast<DefinedRegular>(b1);
    auto *d2 = dyn_cast<DefinedRegular>(b2);
    if (!d1 || !d2)
      return false;
    if constexpr (constant)
      if (d1->getValue() != d2->getValue())
        return false;
    return d1->getChunk()->eqClass[cnt % 2] == d2->getChunk()->eqClass[cnt % 2];
  };

  auto eqReloc = [&](const coff_relocation &r1, const coff_relocation &r2) {
    if constexpr (constant)
      if (r1.Type != r2.Type || r1.VirtualAddress != r2.VirtualAddress)
        return false;
    return eqSym(a->file->getSymbol(r1.SymbolTableIndex),
                 b->file->getSymbol(r2.SymbolTableIndex));
  };
  if (!llvm::equal(a->getRelocs(), b->getRelocs(), eqReloc))
    return false;

  if constexpr (constant) {
    return a->getOutputCharacteristics() == b->getOutputCharacteristics() &&
           a->getSectionName() == b->getSectionName() &&
           a->header->SizeOfRawData == b->header->SizeOfRawData &&
           a->checksum == b->checksum && a->getContents() == b->getContents() &&
           a->getMachine() == b->getMachine() && assocEquals(a, b);
  } else {
    Symbol *e1 = a->getEntryThunk();
    Symbol *e2 = b->getEntryThunk();
    if ((e1 || e2) && (!e1 || !e2 || !eqSym(e1, e2)))
      return false;

    // Check associated children sections, i.e. exception handling data, for
    // equality.
    return assocEquals(a, b);
  }
}

// Find the first Chunk after Begin that has a different class from Begin.
size_t ICF::findBoundary(size_t begin, size_t end) {
  for (size_t i = begin + 1; i < end; ++i)
    if (chunks[begin]->eqClass[cnt % 2] != chunks[i]->eqClass[cnt % 2])
      return i;
  return end;
}

void ICF::forEachClassRange(size_t begin, size_t end,
                            std::function<void(size_t, size_t)> fn) {
  while (begin < end) {
    size_t mid = findBoundary(begin, end);
    fn(begin, mid);
    begin = mid;
  }
}

// Call Fn on each class group.
void ICF::forEachClass(std::function<void(size_t, size_t)> fn) {
  // If the number of sections are too small to use threading,
  // call Fn sequentially.
  if (chunks.size() < 1024) {
    forEachClassRange(0, chunks.size(), fn);
    ++cnt;
    return;
  }

  // Shard into non-overlapping intervals, and call Fn in parallel.
  // The sharding must be completed before any calls to Fn are made
  // so that Fn can modify the Chunks in its shard without causing data
  // races.
  const size_t numShards = 256;
  size_t step = chunks.size() / numShards;
  size_t boundaries[numShards + 1];
  boundaries[0] = 0;
  boundaries[numShards] = chunks.size();
  parallelFor(1, numShards, [&](size_t i) {
    boundaries[i] = findBoundary((i - 1) * step, chunks.size());
  });
  parallelFor(1, numShards + 1, [&](size_t i) {
    if (boundaries[i - 1] < boundaries[i]) {
      forEachClassRange(boundaries[i - 1], boundaries[i], fn);
    }
  });
  ++cnt;
}

// Merge identical COMDAT sections.
// Two sections are considered the same if their section headers,
// contents and relocations are all the same.
void ICF::run() {
  llvm::TimeTraceScope timeScope("ICF");
  ScopedTimer t(ctx.icfTimer);

  // Collect only mergeable sections and group by hash value.
  uint32_t nextId = 1;
  for (Chunk *c : ctx.driver.getChunks()) {
    if (auto *sc = dyn_cast<SectionChunk>(c)) {
      if (isEligible(sc)) {
        // Candidates hold their index, tagged with the MSB, until hashing
        // completes.
        sc->eqClass[0] = chunks.size() | (1U << 31);
        chunks.push_back(sc);
      } else {
        sc->eqClass[0] = nextId++;
      }
    }
  }

  // Make sure that ICF doesn't merge sections that are being handled by string
  // tail merging.
  for (MergeChunk *mc : ctx.mergeChunkInstances)
    if (mc)
      for (SectionChunk *sc : mc->sections)
        sc->eqClass[0] = nextId++;

  assert(chunks.size() <= (1U << 31) && nextId < (1U << 31) &&
         "the MSB marks a candidate and the low bits index chunks");

  // Hash sections in chunks. An edge is a relocation or associated section
  // referencing a candidate, recorded as the target's index. Each chunk
  // appends its sections' edges to its own buffer; edgeEnd[i] is where
  // section i's edges end within that buffer.
  constexpr size_t blockSize = 1024;
  size_t numChunks = chunks.size();
  size_t numBlocks = divideCeil(numChunks, blockSize);
  SmallVector<uint64_t, 0> hashes, nextHashes;
  hashes.resize_for_overwrite(numChunks);
  nextHashes.resize_for_overwrite(numChunks);
  SmallVector<uint32_t, 0> edgeEnd;
  edgeEnd.resize_for_overwrite(numChunks);
  SmallVector<SmallVector<uint32_t, 0>, 0> chunkEdges(numBlocks);
  parallelFor(0, numBlocks, [&](size_t b) {
    SmallVectorImpl<uint32_t> &edges = chunkEdges[b];
    size_t begin = b * blockSize, end = std::min(begin + blockSize, numChunks);
    for (size_t i = begin; i != end; ++i) {
      SectionChunk *sc = chunks[i];
      uint64_t h = xxh3_64bits(sc->getContents());
      h = mix(h, sc->getOutputCharacteristics());
      h = mix(h, xxh3_64bits(sc->getSectionName()));
      h = mix(h, sc->header->SizeOfRawData);
      h = mix(h, sc->checksum);
      h = mix(h, sc->getMachine());
      auto hashChunk = [&](SectionChunk *target) {
        if (target->eqClass[0] & (1U << 31))
          edges.push_back(target->eqClass[0] & ~(1U << 31));
        else
          h = mix(h, target->eqClass[0]);
      };
      for (const coff_relocation &r : sc->getRelocs()) {
        h = mix(h, r.Type);
        h = mix(h, r.VirtualAddress);
        Symbol *sym = sc->file->getSymbol(r.SymbolTableIndex);
        if (auto *d = dyn_cast_or_null<DefinedRegular>(sym)) {
          h = mix(h, d->getValue());
          hashChunk(d->getChunk());
        } else {
          h = mix(h, sym ? xxh3_64bits(sym->getName()) : 0);
        }
      }
      if (Defined *e = sc->getEntryThunk()) {
        if (auto *d = dyn_cast<DefinedRegular>(e))
          hashChunk(d->getChunk());
        else
          h = mix(h, xxh3_64bits(e->getName()));
      }
      size_t numAssoc = 0;
      for (const SectionChunk &c : sc->children()) {
        if (!considerForICF(c))
          continue;
        ++numAssoc;
        hashChunk(const_cast<SectionChunk *>(&c));
      }
      h = mix(h, numAssoc);
      hashes[i] = h;
      edgeEnd[i] = edges.size();
    }
  });

  // Propagate hashes over edges for 7 rounds. Each round saves up to one
  // segregate() pass. 7 is empirical: even Chromium then needs just two passes.
  for (int round = 7; round--;) {
    parallelFor(0, numBlocks, [&](size_t b) {
      ArrayRef<uint32_t> edges = chunkEdges[b];
      size_t begin = b * blockSize,
             end = std::min(begin + blockSize, numChunks), j = 0;
      for (size_t i = begin; i != end; ++i) {
        uint64_t h = hashes[i];
        for (size_t e = edgeEnd[i]; j != e; ++j)
          h = mix(h, hashes[edges[j]]);
        nextHashes[i] = h;
      }
    });
    std::swap(hashes, nextHashes);
  }

  // From now on, sections in Chunks are ordered so that sections in
  // the same group are consecutive in the vector.
  SmallVector<uint64_t, 0> keys;
  keys.resize_for_overwrite(numChunks);
  parallelFor(0, numChunks, [&](size_t i) {
    // Set MSB to 1 to avoid collisions with non-hash classes.
    uint32_t h = hashes[i] | (1U << 31);
    chunks[i]->eqClass[0] = h;
    keys[i] = uint64_t(h) << 32 | i;
  });
  parallelSort(keys.begin(), keys.end());
  std::vector<SectionChunk *> sorted(numChunks);
  parallelFor(0, numChunks,
              [&](size_t i) { sorted[i] = chunks[uint32_t(keys[i])]; });
  chunks = std::move(sorted);

  // Compare static contents and assign unique IDs for each static content.
  forEachClass([&](size_t begin, size_t end) { segregate<true>(begin, end); });

  // Split groups by comparing relocations until convergence is obtained.
  do {
    repeat = false;
    forEachClass(
        [&](size_t begin, size_t end) { segregate<false>(begin, end); });
  } while (repeat);

  Log(ctx) << "ICF needed " << Twine(cnt) << " iterations";

  // Merge sections in the same classes.
  forEachClass([&](size_t begin, size_t end) {
    if (end - begin == 1)
      return;

    Log(ctx) << "Selected " << chunks[begin]->getDebugName();
    for (size_t i = begin + 1; i < end; ++i) {
      Log(ctx) << "  Removed " << chunks[i]->getDebugName();
      chunks[begin]->replace(chunks[i]);
    }
  });
}

// Entry point to ICF.
void doICF(COFFLinkerContext &ctx) { ICF(ctx).run(); }

} // namespace lld::coff
