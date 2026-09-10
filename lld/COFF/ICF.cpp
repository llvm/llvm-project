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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <atomic>
#include <vector>

using namespace llvm;

namespace lld::coff {

class ICF {
public:
  ICF(COFFLinkerContext &c) : ctx(c){};
  void run();

private:
  void recordPdataRefs();
  void segregate(size_t begin, size_t end, bool constant);

  bool sectionEquals(const SectionChunk *a, const SectionChunk *b,
                     bool constant);
  bool pdataEquals(const SectionChunk *a, const SectionChunk *b,
                   bool constant);
  bool describedPdataEquals(const SectionChunk *a, const SectionChunk *b,
                            bool constant);
  bool assocEquals(const SectionChunk *a, const SectionChunk *b,
                   bool constant);

  bool equalsConstant(const SectionChunk *a, const SectionChunk *b);
  bool equalsVariable(const SectionChunk *a, const SectionChunk *b);

  bool isEligible(SectionChunk *c);

  size_t findBoundary(size_t begin, size_t end);

  void forEachClassRange(size_t begin, size_t end,
                         std::function<void(size_t, size_t)> fn);

  void forEachClass(std::function<void(size_t, size_t)> fn);

  std::vector<SectionChunk *> chunks;
  DenseMap<const SectionChunk *, SmallVector<SectionChunk *, 1>> pdataRefs;
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

static SectionChunk *getSectionChunk(Symbol *sym) {
  if (auto *d = dyn_cast_or_null<Defined>(sym))
    return dyn_cast_or_null<SectionChunk>(d->getChunk());
  return nullptr;
}

static bool isPDataMachine(MachineTypes machine) {
  return machine == AMD64 || machine == ARM64;
}

static size_t getPDataSize(MachineTypes machine) {
  if (machine == ARM64)
    return 8;
  return 12;
}

static bool isPDataReloc(const coff_relocation &rel, MachineTypes machine) {
  bool addr32 = machine == ARM64
                    ? rel.Type == llvm::COFF::IMAGE_REL_ARM64_ADDR32 ||
                          rel.Type == llvm::COFF::IMAGE_REL_ARM64_ADDR32NB
                    : rel.Type == llvm::COFF::IMAGE_REL_AMD64_ADDR32 ||
                          rel.Type == llvm::COFF::IMAGE_REL_AMD64_ADDR32NB;
  return addr32 && rel.VirtualAddress < getPDataSize(machine) &&
         rel.VirtualAddress % 4 == 0;
}

static const coff_relocation *findRelocAt(ArrayRef<coff_relocation> relocs,
                                          uint32_t virtualAddress) {
  for (const coff_relocation &rel : relocs)
    if (rel.VirtualAddress == virtualAddress)
      return &rel;
  return nullptr;
}

static bool isPData(const SectionChunk *sc) {
  return sc && isPDataMachine(sc->getMachine()) &&
         sc->getSectionName().split('$').first == ".pdata";
}

void ICF::recordPdataRefs() {
  for (Chunk *c : ctx.driver.getChunks()) {
    auto *sc = dyn_cast<SectionChunk>(c);
    if (!sc || !sc->live || !isPData(sc))
      continue;

    MachineTypes machine = sc->getMachine();

    ArrayRef<coff_relocation> relocs = sc->getRelocs();
    if (sc->getContents().size() != getPDataSize(machine))
      continue;

    const coff_relocation *beginRel = findRelocAt(relocs, 0);
    if (!beginRel || !llvm::all_of(relocs, [&](const coff_relocation &rel) {
          return isPDataReloc(rel, machine);
        }))
      continue;

    SectionChunk *begin =
        getSectionChunk(sc->file->getSymbol(beginRel->SymbolTableIndex));
    if (!begin || !(begin->getOutputCharacteristics() &
                    llvm::COFF::IMAGE_SCN_MEM_EXECUTE))
      continue;

    pdataRefs[begin].push_back(sc);
  }

  for (auto &it : pdataRefs) {
    llvm::stable_sort(it.second, [](const SectionChunk *a,
                                    const SectionChunk *b) {
      if (a->getSectionName() != b->getSectionName())
        return a->getSectionName() < b->getSectionName();
      ArrayRef<uint8_t> ac = a->getContents();
      ArrayRef<uint8_t> bc = b->getContents();
      if (!std::equal(ac.begin(), ac.end(), bc.begin(), bc.end()))
        return std::lexicographical_compare(ac.begin(), ac.end(), bc.begin(),
                                            bc.end());
      return a < b;
    });
  }
}

// Split an equivalence class into smaller classes.
void ICF::segregate(size_t begin, size_t end, bool constant) {
  while (begin < end) {
    // Divide [Begin, End) into two. Let Mid be the start index of the
    // second group.
    auto bound = std::stable_partition(
        chunks.begin() + begin + 1, chunks.begin() + end, [&](SectionChunk *s) {
          if (constant)
            return equalsConstant(chunks[begin], s);
          return equalsVariable(chunks[begin], s);
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

bool ICF::sectionEquals(const SectionChunk *a, const SectionChunk *b,
                        bool constant) {
  if (constant)
    return equalsConstant(a, b);
  return equalsVariable(a, b);
}

static bool isXData(const SectionChunk *sc) {
  return sc && sc->getSectionName().split('$').first == ".xdata";
}

static bool sectionNamesEqual(const SectionChunk *a, const SectionChunk *b) {
  if (isXData(a) || isXData(b))
    return isXData(a) && isXData(b) &&
           a->getSectionName().split('$').first ==
               b->getSectionName().split('$').first;
  return a->getSectionName() == b->getSectionName();
}

// On AMD64 and ARM64, .pdata contributes runtime-function table entries that
// map code ranges to unwind data used by stack unwinding and EH dispatch. That
// makes .pdata part of a function's semantic identity: two functions with
// identical .text may still behave differently if their .pdata records refer to
// different .xdata. Associative COMDAT children normally make that dependency
// visible to ICF, but old MSVC-produced object files do not always associate
// .xdata with the parent .text section. Follow .pdata relocations defensively so
// folding is allowed only when the reachable unwind info is equivalent.
bool ICF::pdataEquals(const SectionChunk *ap, const SectionChunk *bp,
                      bool constant) {
  if (!isPData(ap) || !isPData(bp) || ap->getMachine() != bp->getMachine())
    return false;

  MachineTypes machine = ap->getMachine();
  if (ap->getOutputCharacteristics() != bp->getOutputCharacteristics() ||
      ap->getSectionName().split('$').first !=
          bp->getSectionName().split('$').first ||
      ap->header->SizeOfRawData != bp->header->SizeOfRawData ||
      ap->checksum != bp->checksum || ap->getContents() != bp->getContents() ||
      ap->getContents().size() != getPDataSize(machine))
    return false;

  ArrayRef<coff_relocation> ar = ap->getRelocs();
  ArrayRef<coff_relocation> br = bp->getRelocs();
  if (ar.size() != br.size())
    return false;

  auto eqSym = [&](Symbol *as, Symbol *bs) {
    if (as == bs)
      return true;

    auto *ad = dyn_cast<DefinedRegular>(as);
    auto *bd = dyn_cast<DefinedRegular>(bs);
    if (!ad || !bd)
      return false;

    SectionChunk *ac = dyn_cast_or_null<SectionChunk>(ad->getChunk());
    SectionChunk *bc = dyn_cast_or_null<SectionChunk>(bd->getChunk());
    if (isXData(ac) || isXData(bc)) {
      if (!ac || !bc || (constant && ad->getValue() != bd->getValue()))
        return false;
      return sectionEquals(ac, bc, constant);
    }

    if (constant && ad->getValue() != bd->getValue())
      return false;
    return ad->getChunk()->eqClass[cnt % 2] ==
           bd->getChunk()->eqClass[cnt % 2];
  };

  for (size_t i = 0; i != ar.size(); ++i) {
    if (!isPDataReloc(ar[i], machine) || ar[i].Type != br[i].Type ||
        ar[i].VirtualAddress != br[i].VirtualAddress)
      return false;

    Symbol *as = ap->file->getSymbol(ar[i].SymbolTableIndex);
    Symbol *bs = bp->file->getSymbol(br[i].SymbolTableIndex);
    if (!eqSym(as, bs))
      return false;
  }

  return true;
}

bool ICF::describedPdataEquals(const SectionChunk *a, const SectionChunk *b,
                               bool constant) {
  auto ai = pdataRefs.find(a);
  auto bi = pdataRefs.find(b);
  if (ai == pdataRefs.end() || bi == pdataRefs.end())
    return ai == pdataRefs.end() && bi == pdataRefs.end();

  ArrayRef<SectionChunk *> ar = ai->second;
  ArrayRef<SectionChunk *> br = bi->second;
  if (ar.size() != br.size())
    return false;

  return std::equal(ar.begin(), ar.end(), br.begin(), br.end(),
                    [&](const SectionChunk *ap, const SectionChunk *bp) {
                      return pdataEquals(ap, bp, constant);
                    });
}

// Returns true if two sections' associative children are equal.
bool ICF::assocEquals(const SectionChunk *a, const SectionChunk *b,
                      bool constant) {
  // Ignore associated metadata sections that don't participate in ICF, such as
  // debug info and CFGuard metadata.
  auto considerForICF = [](const SectionChunk &assoc) {
    StringRef Name = assoc.getSectionName();
    return !(Name.starts_with(".debug") || Name == ".gfids$y" ||
             Name == ".giats$y" || Name == ".gljmp$y");
  };
  auto ra = make_filter_range(a->children(), considerForICF);
  auto rb = make_filter_range(b->children(), considerForICF);
  return std::equal(ra.begin(), ra.end(), rb.begin(), rb.end(),
                    [&](const SectionChunk &ia, const SectionChunk &ib) {
                      return sectionEquals(&ia, &ib, constant);
                    });
}

// Compare "non-moving" part of two sections, namely everything
// except relocation targets.
bool ICF::equalsConstant(const SectionChunk *a, const SectionChunk *b) {
  if (a->relocsSize != b->relocsSize)
    return false;

  if (isPData(a) || isPData(b))
    return isPData(a) && isPData(b) && pdataEquals(a, b, true);

  // Compare relocations.
  auto eq = [&](const coff_relocation &r1, const coff_relocation &r2) {
    if (r1.Type != r2.Type ||
        r1.VirtualAddress != r2.VirtualAddress) {
      return false;
    }
    Symbol *b1 = a->file->getSymbol(r1.SymbolTableIndex);
    Symbol *b2 = b->file->getSymbol(r2.SymbolTableIndex);
    if (b1 == b2)
      return true;
    if (auto *d1 = dyn_cast<DefinedRegular>(b1))
      if (auto *d2 = dyn_cast<DefinedRegular>(b2))
        return d1->getValue() == d2->getValue() &&
               d1->getChunk()->eqClass[cnt % 2] ==
                   d2->getChunk()->eqClass[cnt % 2];
    return false;
  };
  if (!std::equal(a->getRelocs().begin(), a->getRelocs().end(),
                  b->getRelocs().begin(), eq))
    return false;

  // Compare section attributes and contents.
  return a->getOutputCharacteristics() == b->getOutputCharacteristics() &&
         sectionNamesEqual(a, b) &&
         a->header->SizeOfRawData == b->header->SizeOfRawData &&
         a->checksum == b->checksum && a->getContents() == b->getContents() &&
         a->getMachine() == b->getMachine() && assocEquals(a, b, true) &&
         describedPdataEquals(a, b, true);
}

// Compare "moving" part of two sections, namely relocation targets.
bool ICF::equalsVariable(const SectionChunk *a, const SectionChunk *b) {
  if (isPData(a) || isPData(b))
    return isPData(a) && isPData(b) && pdataEquals(a, b, false);

  // Compare relocations.
  auto eqSym = [&](Symbol *b1, Symbol *b2) {
    if (b1 == b2)
      return true;
    if (auto *d1 = dyn_cast<DefinedRegular>(b1))
      if (auto *d2 = dyn_cast<DefinedRegular>(b2))
        return d1->getChunk()->eqClass[cnt % 2] ==
               d2->getChunk()->eqClass[cnt % 2];
    return false;
  };
  auto eq = [&](const coff_relocation &r1, const coff_relocation &r2) {
    Symbol *b1 = a->file->getSymbol(r1.SymbolTableIndex);
    Symbol *b2 = b->file->getSymbol(r2.SymbolTableIndex);
    return eqSym(b1, b2);
  };

  Symbol *e1 = a->getEntryThunk();
  Symbol *e2 = b->getEntryThunk();
  if ((e1 || e2) && (!e1 || !e2 || !eqSym(e1, e2)))
    return false;

  return std::equal(a->getRelocs().begin(), a->getRelocs().end(),
                    b->getRelocs().begin(), eq) &&
         assocEquals(a, b, false) && describedPdataEquals(a, b, false);
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

  recordPdataRefs();

  // Collect only mergeable sections and group by hash value.
  uint32_t nextId = 1;
  for (Chunk *c : ctx.driver.getChunks()) {
    if (auto *sc = dyn_cast<SectionChunk>(c)) {
      if (isEligible(sc))
        chunks.push_back(sc);
      else
        sc->eqClass[0] = nextId++;
    }
  }

  // Make sure that ICF doesn't merge sections that are being handled by string
  // tail merging.
  for (MergeChunk *mc : ctx.mergeChunkInstances)
    if (mc)
      for (SectionChunk *sc : mc->sections)
        sc->eqClass[0] = nextId++;

  // Initially, we use hash values to partition sections.
  parallelForEach(chunks, [&](SectionChunk *sc) {
    sc->eqClass[0] = xxh3_64bits(sc->getContents());
  });

  // Combine the hashes of the sections referenced by each section into its
  // hash.
  for (unsigned cnt = 0; cnt != 2; ++cnt) {
    parallelForEach(chunks, [&](SectionChunk *sc) {
      uint32_t hash = sc->eqClass[cnt % 2];
      for (Symbol *b : sc->symbols())
        if (auto *sym = dyn_cast_or_null<DefinedRegular>(b))
          hash += sym->getChunk()->eqClass[cnt % 2];
      // Set MSB to 1 to avoid collisions with non-hash classes.
      sc->eqClass[(cnt + 1) % 2] = hash | (1U << 31);
    });
  }

  // From now on, sections in Chunks are ordered so that sections in
  // the same group are consecutive in the vector.
  llvm::stable_sort(chunks, [](const SectionChunk *a, const SectionChunk *b) {
    return a->eqClass[0] < b->eqClass[0];
  });

  // Compare static contents and assign unique IDs for each static content.
  forEachClass([&](size_t begin, size_t end) { segregate(begin, end, true); });

  // Split groups by comparing relocations until convergence is obtained.
  do {
    repeat = false;
    forEachClass(
        [&](size_t begin, size_t end) { segregate(begin, end, false); });
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
