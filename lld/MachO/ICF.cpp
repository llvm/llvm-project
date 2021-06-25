//===- ICF.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ICF.h"
#include "ConcatOutputSection.h"
#include "InputSection.h"
#include "Symbols.h"
#include "llvm/Support/Parallel.h"

#include <atomic>

using namespace llvm;
using namespace lld;
using namespace lld::macho;

ICF::ICF(std::vector<ConcatInputSection *> &inputs) {
  icfInputs.assign(inputs.begin(), inputs.end());
}

// ICF = Identical Code Folding
//
// We only fold __TEXT,__text, so this is really "code" folding, and not
// "COMDAT" folding. String and scalar constant literals are deduplicated
// elsewhere.
//
// Summary of segments & sections:
//
// Since folding never occurs across output-section boundaries,
// ConcatOutputSection is the natural input for ICF.
//
// The __TEXT segment is readonly at the MMU. Some sections are already
// deduplicated elsewhere (__TEXT,__cstring & __TEXT,__literal*) and some are
// synthetic and inherently free of duplicates (__TEXT,__stubs &
// __TEXT,__unwind_info). We only run ICF on __TEXT,__text. One might hope ICF
// could work on __TEXT,__concat, but doing so induces many test failures.
//
// The __LINKEDIT segment is readonly at the MMU, yet entirely synthetic, and
// thus ineligible for ICF.
//
// The __DATA_CONST segment is read/write at the MMU, but is logically const to
// the application after dyld applies fixups to pointer data. Some sections are
// deduplicated elsewhere (__DATA_CONST,__cfstring), and some are synthetic
// (__DATA_CONST,__got). There are no ICF opportunities here.
//
// The __DATA segment is read/write at the MMU, and as application-writeable
// data, none of its sections are eligible for ICF.
//
// Please see the large block comment in lld/ELF/ICF.cpp for an explanation
// of the segregation algorithm.
//
// FIXME(gkm): implement keep-unique attributes
// FIXME(gkm): implement address-significance tables for MachO object files

static unsigned icfPass = 0;
static std::atomic<bool> icfRepeat{false};

// Compare everything except the relocation referents
static bool equalsConstant(const ConcatInputSection *ia,
                           const ConcatInputSection *ib) {
  if (ia->data.size() != ib->data.size())
    return false;
  if (ia->data != ib->data)
    return false;
  if (ia->flags != ib->flags)
    return false;
  if (ia->relocs.size() != ib->relocs.size())
    return false;
  auto f = [&](const Reloc &ra, const Reloc &rb) {
    if (ra.type != rb.type)
      return false;
    if (ra.pcrel != rb.pcrel)
      return false;
    if (ra.length != rb.length)
      return false;
    if (ra.offset != rb.offset)
      return false;
    if (ra.addend != rb.addend)
      return false;
    if (ra.referent.is<Symbol *>() != rb.referent.is<Symbol *>())
      return false; // a nice place to breakpoint
    return true;
  };
  return std::equal(ia->relocs.begin(), ia->relocs.end(), ib->relocs.begin(),
                    f);
}

// Compare only the relocation referents
static bool equalsVariable(const ConcatInputSection *ia,
                           const ConcatInputSection *ib) {
  assert(ia->relocs.size() == ib->relocs.size());
  auto f = [&](const Reloc &ra, const Reloc &rb) {
    if (ra.referent == rb.referent)
      return true;
    if (ra.referent.is<Symbol *>()) {
      const auto *sa = ra.referent.get<Symbol *>();
      const auto *sb = rb.referent.get<Symbol *>();
      if (sa->kind() != sb->kind())
        return false;
      if (isa<Defined>(sa)) {
        const auto *da = dyn_cast<Defined>(sa);
        const auto *db = dyn_cast<Defined>(sb);
        if (da->value != db->value)
          return false;
        if (da->isAbsolute() != db->isAbsolute())
          return false;
        if (da->isec) {
          if (da->isec->kind() != db->isec->kind())
            return false;
          if (const auto *isecA = dyn_cast<ConcatInputSection>(da->isec)) {
            const auto *isecB = cast<ConcatInputSection>(db->isec);
            if (isecA->icfEqClass[icfPass % 2] !=
                isecB->icfEqClass[icfPass % 2])
              return false;
          } else {
            // FIXME: implement ICF for other InputSection kinds
            return false;
          }
        }
      } else if (isa<DylibSymbol>(sa)) {
        // There is one DylibSymbol per gotIndex and we already checked for
        // symbol equality, thus we know that these must be different.
        return false;
      } else {
        llvm_unreachable("equalsVariable symbol kind");
      }
    } else {
      const auto *sa = ra.referent.get<InputSection *>();
      const auto *sb = rb.referent.get<InputSection *>();
      if (sa->kind() != sb->kind())
        return false;
      if (const auto *isecA = dyn_cast<ConcatInputSection>(sa)) {
        const auto *isecB = cast<ConcatInputSection>(sb);
        if (isecA->icfEqClass[icfPass % 2] != isecB->icfEqClass[icfPass % 2])
          return false;
      } else {
        // FIXME: implement ICF for other InputSection kinds
        return false;
      }
    }
    return true;
  };
  return std::equal(ia->relocs.begin(), ia->relocs.end(), ib->relocs.begin(),
                    f);
}

// Find the first InputSection after BEGIN whose equivalence class differs
size_t ICF::findBoundary(size_t begin, size_t end) {
  uint64_t beginHash = icfInputs[begin]->icfEqClass[icfPass % 2];
  for (size_t i = begin + 1; i < end; ++i)
    if (beginHash != icfInputs[i]->icfEqClass[icfPass % 2])
      return i;
  return end;
}

// Invoke FUNC on subranges with matching equivalence class
void ICF::forEachClassRange(size_t begin, size_t end,
                            std::function<void(size_t, size_t)> func) {
  while (begin < end) {
    size_t mid = findBoundary(begin, end);
    func(begin, mid);
    begin = mid;
  }
}

// Split icfInputs into shards, then parallelize invocation of FUNC on subranges
// with matching equivalence class
void ICF::forEachClass(std::function<void(size_t, size_t)> func) {
  // Only use threads when the benefits outweigh the overhead.
  const size_t threadingThreshold = 1024;
  if (icfInputs.size() < threadingThreshold) {
    forEachClassRange(0, icfInputs.size(), func);
    ++icfPass;
    return;
  }

  // Shard into non-overlapping intervals, and call FUNC in parallel.  The
  // sharding must be completed before any calls to FUNC are made so that FUNC
  // can modify the InputSection in its shard without causing data races.
  const size_t shards = 256;
  size_t step = icfInputs.size() / shards;
  size_t boundaries[shards + 1];
  boundaries[0] = 0;
  boundaries[shards] = icfInputs.size();
  parallelForEachN(1, shards, [&](size_t i) {
    boundaries[i] = findBoundary((i - 1) * step, icfInputs.size());
  });
  parallelForEachN(1, shards + 1, [&](size_t i) {
    if (boundaries[i - 1] < boundaries[i]) {
      forEachClassRange(boundaries[i - 1], boundaries[i], func);
    }
  });
  ++icfPass;
}

void ICF::run() {
  // Into each origin-section hash, combine all reloc referent section hashes.
  for (icfPass = 0; icfPass < 2; ++icfPass) {
    parallelForEach(icfInputs, [&](ConcatInputSection *isec) {
      uint64_t hash = isec->icfEqClass[icfPass % 2];
      for (const Reloc &r : isec->relocs) {
        if (auto *sym = r.referent.dyn_cast<Symbol *>()) {
          if (auto *dylibSym = dyn_cast<DylibSymbol>(sym))
            hash += dylibSym->stubsHelperIndex;
          else if (auto *defined = dyn_cast<Defined>(sym)) {
            hash += defined->value;
            if (defined->isec)
              if (auto *isec = cast<ConcatInputSection>(defined->isec))
                hash += isec->icfEqClass[icfPass % 2];
            // FIXME: implement ICF for other InputSection kinds
          } else
            llvm_unreachable("foldIdenticalSections symbol kind");
        }
      }
      // Set MSB to 1 to avoid collisions with non-hashed classes.
      isec->icfEqClass[(icfPass + 1) % 2] = hash | (1ull << 63);
    });
  }

  llvm::stable_sort(
      icfInputs, [](const ConcatInputSection *a, const ConcatInputSection *b) {
        return a->icfEqClass[0] < b->icfEqClass[0];
      });
  forEachClass(
      [&](size_t begin, size_t end) { segregate(begin, end, equalsConstant); });

  // Split equivalence groups by comparing relocations until convergence
  do {
    icfRepeat = false;
    forEachClass([&](size_t begin, size_t end) {
      segregate(begin, end, equalsVariable);
    });
  } while (icfRepeat);
  log("ICF needed " + Twine(icfPass) + " iterations");

  // Fold sections within equivalence classes
  forEachClass([&](size_t begin, size_t end) {
    if (end - begin < 2)
      return;
    ConcatInputSection *beginIsec = icfInputs[begin];
    for (size_t i = begin + 1; i < end; ++i)
      beginIsec->foldIdentical(icfInputs[i]);
  });
}

// Split an equivalence class into smaller classes.
void ICF::segregate(
    size_t begin, size_t end,
    std::function<bool(const ConcatInputSection *, const ConcatInputSection *)>
        equals) {
  while (begin < end) {
    // Divide [begin, end) into two. Let mid be the start index of the
    // second group.
    auto bound = std::stable_partition(icfInputs.begin() + begin + 1,
                                       icfInputs.begin() + end,
                                       [&](ConcatInputSection *isec) {
                                         return equals(icfInputs[begin], isec);
                                       });
    size_t mid = bound - icfInputs.begin();

    // Split [begin, end) into [begin, mid) and [mid, end). We use mid as an
    // equivalence class ID because every group ends with a unique index.
    for (size_t i = begin; i < mid; ++i)
      icfInputs[i]->icfEqClass[(icfPass + 1) % 2] = mid;

    // If we created a group, we need to iterate the main loop again.
    if (mid != end)
      icfRepeat = true;

    begin = mid;
  }
}
