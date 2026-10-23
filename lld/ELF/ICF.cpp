//===- ICF.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ICF is short for Identical Code Folding. This is a size optimization to
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
// thus you know they are mergeable, but how do you know they are
// mergeable in the first place? This is not an easy problem to solve.
//
// What we are doing in LLD is to partition sections into equivalence
// classes. Sections in the same equivalence class when the algorithm
// terminates are considered identical. Here are details:
//
// 1. First, we partition sections using their hash values as keys. The hash
//    covers section contents and relocations. A relocation to another ICF
//    candidate contributes the target's hash, propagated for a few rounds,
//    so the hash also covers sections a few relocation hops away.
//
// 2. Next, we split each equivalence class by comparing everything except
//    relocation targets (equalsConstant). These properties never change, so
//    this step runs once.
//
// 3. Then we split classes by comparing relocation targets (equalsVariable),
//    which are considered equivalent if they are in the same equivalence
//    class. A split may make two relocations that targeted the same class
//    target different classes, so we repeat this step until no class splits.
//
// 4. For each equivalence class C, pick an arbitrary section in C, and
//    merge all the other sections in C with it.
//
// Step 1 must give identical sections the same hash. A collision only costs
// step 2 time, and the more relocation hops step 1 covers, the fewer
// iterations step 3 needs.
//
// We parallelize each step so that multiple threads can work on different
// equivalence classes concurrently.
//
//===----------------------------------------------------------------------===//

#include "ICF.h"
#include "Config.h"
#include "InputFiles.h"
#include "LinkerScript.h"
#include "OutputSections.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"
#include "llvm/Support/xxhash.h"
#include <algorithm>
#include <atomic>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace lld;
using namespace lld::elf;

namespace {
template <class ELFT> class ICF {
public:
  ICF(Ctx &ctx) : ctx(ctx) {}
  void run();

private:
  void segregate(size_t begin, size_t end, uint32_t eqClassBase, bool constant);

  template <class RelTy>
  bool constantEq(const InputSection *a, Relocs<RelTy> relsA,
                  const InputSection *b, Relocs<RelTy> relsB);

  template <class RelTy>
  void hashRelocs(const InputSection &sec, uint64_t &h,
                  SmallVectorImpl<uint32_t> &edges, uint32_t eqClassBase,
                  Relocs<RelTy> rels);

  template <class RelTy>
  bool variableEq(const InputSection *a, Relocs<RelTy> relsA,
                  const InputSection *b, Relocs<RelTy> relsB);

  bool equalsConstant(const InputSection *a, const InputSection *b);
  bool equalsVariable(const InputSection *a, const InputSection *b);

  size_t findBoundary(size_t begin, size_t end);

  void forEachClassRange(size_t begin, size_t end,
                         llvm::function_ref<void(size_t, size_t)> fn);

  void parallelForEachClass(llvm::function_ref<void(size_t, size_t)> fn);

  Ctx &ctx;
  SmallVector<InputSection *, 0> sections;

  // We repeat the main loop while `Repeat` is true.
  std::atomic<bool> repeat;

  // The main loop counter.
  int cnt = 0;

  // We have two locations for equivalence classes. On the first iteration
  // of the main loop, Class[0] has a valid value, and Class[1] contains
  // garbage. We read equivalence classes from slot 0 and write to slot 1.
  // So, Class[0] represents the current class, and Class[1] represents
  // the next class. On each iteration, we switch their roles and use them
  // alternately.
  //
  // Why are we doing this? Recall that other threads may be working on
  // other equivalence classes in parallel. They may read sections that we
  // are updating. We cannot update equivalence classes in place because
  // it breaks the invariance that all possibly-identical sections must be
  // in the same equivalence class at any moment. In other words, the for
  // loop to update equivalence classes is not atomic, and that is
  // observable from other threads. By writing new classes to other
  // places, we can keep the invariance.
  //
  // Below, `Current` has the index of the current class, and `Next` has
  // the index of the next class. If threading is enabled, they are either
  // (0, 1) or (1, 0).
  //
  // Note on single-thread: if that's the case, they are always (0, 0)
  // because we can safely read the next class without worrying about race
  // conditions. Using the same location makes this algorithm converge
  // faster because it uses results of the same iteration earlier.
  int current = 0;
  int next = 0;
};
} // namespace

// Returns true if section S is subject of ICF.
static bool isEligible(InputSection *s) {
  if (!s->isLive() || s->keepUnique || !(s->flags & SHF_ALLOC))
    return false;

  // Don't merge writable sections. .data.rel.ro sections are marked as writable
  // but are semantically read-only.
  if ((s->flags & SHF_WRITE) && s->name != ".data.rel.ro" &&
      !s->name.starts_with(".data.rel.ro."))
    return false;

  // SHF_LINK_ORDER sections are ICF'd as a unit with their dependent sections,
  // so we don't consider them for ICF individually.
  if (s->flags & SHF_LINK_ORDER)
    return false;

  // Don't merge synthetic sections as their Data member is not valid and empty.
  // The Data member needs to be valid for ICF as it is used by ICF to determine
  // the equality of section contents.
  if (isa<SyntheticSection>(s))
    return false;

  // .init and .fini contains instructions that must be executed to initialize
  // and finalize the process. They cannot and should not be merged.
  if (s->name == ".init" || s->name == ".fini")
    return false;

  // A user program may enumerate sections named with a C identifier using
  // __start_* and __stop_* symbols. We cannot ICF any such sections because
  // that could change program semantics.
  if (isValidCIdentifier(s->name))
    return false;

  return true;
}

// Split an equivalence class into smaller classes.
template <class ELFT>
void ICF<ELFT>::segregate(size_t begin, size_t end, uint32_t eqClassBase,
                          bool constant) {
  // This loop rearranges sections in [Begin, End) so that all sections
  // that are equal in terms of equals{Constant,Variable} are contiguous
  // in [Begin, End).
  //
  // The algorithm is quadratic in the worst case, but that is not an
  // issue in practice because the number of the distinct sections in
  // each range is usually very small.

  while (begin < end) {
    // Divide [Begin, End) into two. Let Mid be the start index of the
    // second group.
    auto bound =
        std::stable_partition(sections.begin() + begin + 1,
                              sections.begin() + end, [&](InputSection *s) {
                                if (constant)
                                  return equalsConstant(sections[begin], s);
                                return equalsVariable(sections[begin], s);
                              });
    size_t mid = bound - sections.begin();

    // Now we split [Begin, End) into [Begin, Mid) and [Mid, End) by
    // updating the sections in [Begin, Mid). We use Mid as the basis for
    // the equivalence class ID because every group ends with a unique index.
    // Add this to eqClassBase to avoid equality with unique IDs.
    for (size_t i = begin; i < mid; ++i)
      sections[i]->eqClass[next] = eqClassBase + mid;

    // If we created a group, we need to iterate the main loop again.
    if (mid != end)
      repeat = true;

    begin = mid;
  }
}

// Propagation mixes a self-referencing section's hash with itself, so mix(h, h)
// must not be a constant.
static uint64_t mix(uint64_t h, uint64_t v) {
  return (h + v) * 0x9e3779b97f4a7c15;
}

// Different symbols compare equal only if both values are final: not a linker
// script placeholder (assigned later) and not preemptible.
static Defined *getComparableDef(Symbol &sym) {
  auto *d = dyn_cast<Defined>(&sym);
  return d && !d->scriptDefined && !d->isPreemptible ? d : nullptr;
}

// A section symbol's addend is not validated until getSymVA. Return nullopt
// to conservatively disable folding for such an out-of-range offset.
static std::optional<uint64_t> getMergeOffset(const Defined &d,
                                              uint64_t addend) {
  auto &ms = cast<MergeInputSection>(*d.section);
  if (!d.isSection())
    return ms.getOffset(d.value) + addend;
  if (addend >= ms.content().size())
    return std::nullopt;
  return ms.getOffset(addend);
}

// Compare two lists of relocations.
template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::constantEq(const InputSection *secA, Relocs<RelTy> ra,
                           const InputSection *secB, Relocs<RelTy> rb) {
  if (ra.size() != rb.size())
    return false;
  auto rai = ra.begin(), rae = ra.end(), rbi = rb.begin();
  for (; rai != rae; ++rai, ++rbi) {
    if (rai->r_offset != rbi->r_offset ||
        rai->getType(ctx.arg.isMips64EL) != rbi->getType(ctx.arg.isMips64EL))
      return false;

    uint64_t addA = getAddend<ELFT>(*rai);
    uint64_t addB = getAddend<ELFT>(*rbi);

    Symbol &sa = secA->file->getRelocTargetSym(*rai);
    Symbol &sb = secB->file->getRelocTargetSym(*rbi);
    if (&sa == &sb && addA == addB)
      continue;

    Defined *da = getComparableDef(sa), *db = getComparableDef(sb);
    if (!da || !db)
      return false;

    // Relocations referring to absolute symbols are constant-equal if their
    // values are equal.
    if (!da->section && !db->section && da->value + addA == db->value + addB)
      continue;
    if (!da->section || !db->section)
      return false;

    if (da->section->kind() != db->section->kind())
      return false;

    // Relocations referring to InputSections are constant-equal if their
    // section offsets are equal.
    if (isa<InputSection>(da->section)) {
      if (da->value + addA == db->value + addB)
        continue;
      return false;
    }

    // Relocations referring to MergeInputSections are constant-equal if their
    // offsets in the output section are equal.
    auto *x = dyn_cast<MergeInputSection>(da->section);
    if (!x)
      return false;
    auto *y = cast<MergeInputSection>(db->section);
    if (x->getParent() != y->getParent())
      return false;

    auto offA = getMergeOffset(*da, addA);
    if (!offA || offA != getMergeOffset(*db, addB))
      return false;
  }

  return true;
}

// Hash what constantEq compares, plus the class of a non-candidate target,
// which never changes. Candidate targets are recorded as edges instead.
template <class ELFT>
template <class RelTy>
void ICF<ELFT>::hashRelocs(const InputSection &sec, uint64_t &h,
                           SmallVectorImpl<uint32_t> &edges,
                           uint32_t eqClassBase, Relocs<RelTy> rels) {
  for (const RelTy &rel : rels) {
    h = mix(h, rel.r_offset);
    h = mix(h, rel.getType(ctx.arg.isMips64EL));
    uint64_t addend = getAddend<ELFT>(rel);
    Symbol &sym = sec.file->getRelocTargetSym(rel);
    Defined *d = getComparableDef(sym);
    if (!d) {
      h = mix(h, xxh3_64bits(sym.getName()));
      h = mix(h, addend);
      continue;
    }
    if (isa_and_nonnull<MergeInputSection>(d->section)) {
      h = mix(h, getMergeOffset(*d, addend).value_or(-1));
      continue;
    }
    h = mix(h, d->value + addend);
    auto *is = dyn_cast_or_null<InputSection>(d->section);
    if (!is)
      continue;
    // A candidate's eqClass[0] - eqClassBase is its index in sections.
    if (is->eqClass[0] >= eqClassBase)
      edges.push_back(is->eqClass[0] - eqClassBase);
    else
      h = mix(h, is->eqClass[0]);
  }
}

// Compare "non-moving" part of two InputSections, namely everything
// except relocation targets.
template <class ELFT>
bool ICF<ELFT>::equalsConstant(const InputSection *a, const InputSection *b) {
  if (a->flags != b->flags || a->getSize() != b->getSize() ||
      a->content() != b->content())
    return false;

  // If two sections have different output sections, we cannot merge them.
  assert(a->getParent() && b->getParent());
  if (a->getParent() != b->getParent())
    return false;

  const RelsOrRelas<ELFT> ra = a->template relsOrRelas<ELFT>();
  const RelsOrRelas<ELFT> rb = b->template relsOrRelas<ELFT>();
  if (ra.areRelocsCrel() || rb.areRelocsCrel())
    return constantEq(a, ra.crels, b, rb.crels);
  return ra.areRelocsRel() || rb.areRelocsRel()
             ? constantEq(a, ra.rels, b, rb.rels)
             : constantEq(a, ra.relas, b, rb.relas);
}

// Compare two lists of relocations. Returns true if all pairs of
// relocations point to the same section in terms of ICF.
template <class ELFT>
template <class RelTy>
bool ICF<ELFT>::variableEq(const InputSection *secA, Relocs<RelTy> ra,
                           const InputSection *secB, Relocs<RelTy> rb) {
  assert(ra.size() == rb.size());

  auto rai = ra.begin(), rae = ra.end(), rbi = rb.begin();
  for (; rai != rae; ++rai, ++rbi) {
    // The two sections must be identical.
    Symbol &sa = secA->file->getRelocTargetSym(*rai);
    Symbol &sb = secB->file->getRelocTargetSym(*rbi);
    if (&sa == &sb)
      continue;

    auto *da = cast<Defined>(&sa);
    auto *db = cast<Defined>(&sb);

    // We already dealt with absolute and non-InputSection symbols in
    // constantEq, and for InputSections we have already checked everything
    // except the equivalence class.
    if (!da->section)
      continue;
    auto *x = dyn_cast<InputSection>(da->section);
    if (!x)
      continue;
    auto *y = cast<InputSection>(db->section);

    // Sections that are in the special equivalence class 0, can never be the
    // same in terms of the equivalence class.
    if (x->eqClass[current] == 0)
      return false;
    if (x->eqClass[current] != y->eqClass[current])
      return false;
  };

  return true;
}

// Compare "moving" part of two InputSections, namely relocation targets.
template <class ELFT>
bool ICF<ELFT>::equalsVariable(const InputSection *a, const InputSection *b) {
  const RelsOrRelas<ELFT> ra = a->template relsOrRelas<ELFT>();
  const RelsOrRelas<ELFT> rb = b->template relsOrRelas<ELFT>();
  if (ra.areRelocsCrel() || rb.areRelocsCrel())
    return variableEq(a, ra.crels, b, rb.crels);
  return ra.areRelocsRel() || rb.areRelocsRel()
             ? variableEq(a, ra.rels, b, rb.rels)
             : variableEq(a, ra.relas, b, rb.relas);
}

template <class ELFT> size_t ICF<ELFT>::findBoundary(size_t begin, size_t end) {
  uint32_t eqClass = sections[begin]->eqClass[current];
  for (size_t i = begin + 1; i < end; ++i)
    if (eqClass != sections[i]->eqClass[current])
      return i;
  return end;
}

// Sections in the same equivalence class are contiguous in Sections
// vector. Therefore, Sections vector can be considered as contiguous
// groups of sections, grouped by the class.
//
// This function calls Fn on every group within [Begin, End).
template <class ELFT>
void ICF<ELFT>::forEachClassRange(size_t begin, size_t end,
                                  llvm::function_ref<void(size_t, size_t)> fn) {
  while (begin < end) {
    size_t mid = findBoundary(begin, end);
    fn(begin, mid);
    begin = mid;
  }
}

// Call Fn on each equivalence class.

template <class ELFT>
void ICF<ELFT>::parallelForEachClass(
    llvm::function_ref<void(size_t, size_t)> fn) {
  // If threading is disabled or the number of sections are
  // too small to use threading, call Fn sequentially.
  if (parallel::strategy.ThreadsRequested == 1 || sections.size() < 1024) {
    forEachClassRange(0, sections.size(), fn);
    ++cnt;
    return;
  }

  current = cnt % 2;
  next = (cnt + 1) % 2;

  // Shard into non-overlapping intervals, and call Fn in parallel.
  // The sharding must be completed before any calls to Fn are made
  // so that Fn can modify the Chunks in its shard without causing data
  // races.
  const size_t numShards = 256;
  size_t step = sections.size() / numShards;
  size_t boundaries[numShards + 1];
  boundaries[0] = 0;
  boundaries[numShards] = sections.size();

  parallelFor(1, numShards, [&](size_t i) {
    boundaries[i] = findBoundary((i - 1) * step, sections.size());
  });

  parallelFor(1, numShards + 1, [&](size_t i) {
    if (boundaries[i - 1] < boundaries[i])
      forEachClassRange(boundaries[i - 1], boundaries[i], fn);
  });
  ++cnt;
}

// The main function of ICF.
template <class ELFT> void ICF<ELFT>::run() {
  // Two text sections may have identical content and relocations but different
  // LSDA, e.g. the two functions may have catch blocks of different types. If a
  // text section is referenced by a .eh_frame FDE with LSDA, it is not
  // eligible. This is implemented by iterating over CIE/FDE and setting
  // eqClass[0] to the referenced text section from a live FDE.
  //
  // If two .gcc_except_table have identical semantics (usually identical
  // content with PC-relative encoding), we will lose folding opportunity.
  uint32_t uniqueId = 1;
  ctx.in.ehFrame->iterateFDEWithLSDA<ELFT>(
      [&](InputSection &s) { s.eqClass[0] = s.eqClass[1] = uniqueId++; });

  // Assign a temporary class ID (eqClassBase) to eligible sections and assign
  // unique IDs to ineligible sections. Use eqClassBase as the base for class
  // IDs. Each ineligible section belongs to an equivalence class of its own.
  constexpr size_t chunkSize = 1024;
  ArrayRef<InputSectionBase *> inputs = ctx.inputSections;
  size_t numInputChunks = divideCeil(inputs.size(), chunkSize);
  auto chunkRange = [&](size_t c, size_t total) {
    auto begin = c * chunkSize;
    return std::make_pair(begin, std::min(begin + chunkSize, total));
  };
  uint32_t eqClassBase = uniqueId + inputs.size();
  SmallVector<uint32_t, 0> firstIdx(numInputChunks);
  parallelFor(0, numInputChunks, [&](size_t c) {
    auto [begin, end] = chunkRange(c, inputs.size());
    uint32_t n = 0;
    for (size_t i = begin; i != end; ++i) {
      auto *s = dyn_cast<InputSection>(inputs[i]);
      if (s && s->eqClass[0] == 0) {
        if (isEligible(s)) {
          s->eqClass[0] = eqClassBase;
          ++n;
        } else {
          s->eqClass[0] = s->eqClass[1] = uniqueId + i;
        }
      }
    }
    // Temporarily store the number of sections per chunk.
    firstIdx[c] = n;
  });

  // Compute the total number of eligible sections and each chunk's left index.
  size_t numSections = 0;
  for (size_t c = 0; c != numInputChunks; ++c) {
    auto n = std::exchange(firstIdx[c], numSections);
    numSections += n;
  }
  sections.resize_for_overwrite(numSections);
  // Update temporary class ID above the unique IDs so that hashRelocs() can
  // tell them apart.
  assert(uint64_t(eqClassBase) + numSections <= UINT32_MAX);
  parallelFor(0, numInputChunks, [&](size_t c) {
    auto [begin, end] = chunkRange(c, inputs.size());
    auto idx = firstIdx[c];
    for (size_t i = begin; i != end; ++i) {
      auto *s = dyn_cast<InputSection>(inputs[i]);
      if (s && s->eqClass[0] == eqClassBase) {
        s->eqClass[0] += idx;
        sections[idx++] = s;
      }
    }
  });

  // Hash sections in chunks. An edge is a relocation to a candidate, recorded
  // as the target's index. Each chunk appends its sections' edges to its own
  // buffer; edgeEnd[i] is where section i's edges end within that buffer.
  size_t numChunks = divideCeil(numSections, chunkSize);
  SmallVector<uint64_t, 0> hashes(numSections, llvm::for_overwrite);
  SmallVector<uint32_t, 0> edgeEnd(numSections, llvm::for_overwrite);
  SmallVector<SmallVector<uint32_t, 0>, 0> chunkEdges(numChunks);
  parallelFor(0, numChunks, [&](size_t c) {
    SmallVectorImpl<uint32_t> &edges = chunkEdges[c];
    auto [begin, end] = chunkRange(c, numSections);
    for (size_t i = begin; i != end; ++i) {
      InputSection *s = sections[i];
      uint64_t h = xxh3_64bits(s->content());
      invokeOnRelocs(*s, hashRelocs, *s, h, edges, eqClassBase);
      hashes[i] = h;
      edgeEnd[i] = edges.size();
    }
  });

  // Propagate hashes over edges for 7 rounds. Each round saves up to one
  // segregate() pass. 7 is empirical: even Chromium then needs just two passes.
  SmallVector<uint64_t, 0> nextHashes(numSections, llvm::for_overwrite);
  for (int round = 7; round--;) {
    parallelFor(0, numChunks, [&](size_t c) {
      ArrayRef<uint32_t> edges = chunkEdges[c];
      auto [begin, end] = chunkRange(c, numSections);
      size_t j = 0;
      for (size_t i = begin; i != end; ++i) {
        uint64_t h = hashes[i];
        for (size_t e = edgeEnd[i]; j != e; ++j)
          h = mix(h, hashes[edges[j]]);
        nextHashes[i] = h;
      }
    });
    std::swap(hashes, nextHashes);
  }

  // Order sections so that those in the same equivalence class are consecutive.
  SmallVector<uint64_t, 0> keys(numSections, llvm::for_overwrite);
  parallelFor(0, numSections, [&](size_t i) {
    // mix()'s low bits depend only on the inputs' low bits. Fold the high half.
    uint32_t h = hashes[i] ^ hashes[i] >> 32;
    sections[i]->eqClass[0] = h;
    keys[i] = uint64_t(h) << 32 | i;
  });
  parallelSort(keys.begin(), keys.end());
  SmallVector<InputSection *, 0> sorted(numSections, llvm::for_overwrite);
  parallelFor(0, numSections,
              [&](size_t i) { sorted[i] = sections[uint32_t(keys[i])]; });
  sections = std::move(sorted);

  // Compare static contents and assign unique equivalence class IDs for each
  // static content. Use a base offset for these IDs to ensure no overlap with
  // the unique IDs already assigned.
  parallelForEachClass([&](size_t begin, size_t end) {
    segregate(begin, end, eqClassBase, true);
  });

  // Split groups by comparing relocations until convergence is obtained.
  do {
    repeat = false;
    parallelForEachClass([&](size_t begin, size_t end) {
      segregate(begin, end, eqClassBase, false);
    });
  } while (repeat);

  Log(ctx) << "ICF needed " << cnt << " iterations";

  // Merge sections by the equivalence class.
  parallelForEachClass([&](size_t begin, size_t end) {
    for (size_t i = begin + 1; i < end; ++i) {
      sections[begin]->replace(sections[i]);

      // At this point we know sections merged are fully identical and hence
      // we want to remove duplicate implicit dependencies such as link order
      // and relocation sections.
      for (InputSection *isec : sections[i]->dependentSections)
        isec->markDead();
    }
  });

  // Print classes in input order, independent of the hash.
  if (ctx.arg.printIcfSections) {
    DenseMap<InputSection *, SmallVector<InputSection *, 0>> folded;
    for (InputSectionBase *sec : ctx.inputSections)
      if (auto *s = dyn_cast<InputSection>(sec); s && s->repl != s)
        folded[s->repl].push_back(s);
    for (InputSectionBase *sec : ctx.inputSections) {
      auto it = folded.find(dyn_cast<InputSection>(sec));
      if (it == folded.end())
        continue;
      Msg(ctx) << "selected section " << sec;
      for (InputSection *s : it->second)
        Msg(ctx) << "  removing identical section " << s;
    }
  }

  // Change Defined symbol's section field to the canonical one.
  auto fold = [](Symbol *sym) {
    if (auto *d = dyn_cast<Defined>(sym))
      if (auto *sec = dyn_cast_or_null<InputSection>(d->section))
        if (sec->repl != d->section) {
          d->section = sec->repl;
          d->folded = true;
        }
  };
  parallelForEach(ctx.symtab->getSymbols(), fold);
  parallelForEach(ctx.objectFiles, [&](ELFFileBase *file) {
    for (Symbol *sym : file->getLocalSymbols())
      fold(sym);
  });

  // InputSectionDescription::sections is populated by processSectionCommands().
  // ICF may fold some input sections assigned to output sections. Remove them.
  for (SectionCommand *cmd : ctx.script->sectionCommands)
    if (auto *osd = dyn_cast<OutputDesc>(cmd))
      for (SectionCommand *subCmd : osd->osec.commands)
        if (auto *isd = dyn_cast<InputSectionDescription>(subCmd))
          llvm::erase_if(isd->sections,
                         [](InputSection *isec) { return !isec->isLive(); });
}

// ICF entry point function.
template <class ELFT> void elf::doIcf(Ctx &ctx) {
  llvm::TimeTraceScope timeScope("ICF");
  ICF<ELFT>(ctx).run();
}

template void elf::doIcf<ELF32LE>(Ctx &);
template void elf::doIcf<ELF32BE>(Ctx &);
template void elf::doIcf<ELF64LE>(Ctx &);
template void elf::doIcf<ELF64BE>(Ctx &);
