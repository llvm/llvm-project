//===- MarkLive.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MarkLive.h"
#include "Config.h"
#include "OutputSegment.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "UnwindInfoSection.h"

#include "lld/Common/ErrorHandler.h"
#include "llvm/Support/Parallel.h"
#include "llvm/Support/TimeProfiler.h"

#include <mutex>

namespace lld::macho {

using namespace llvm;
using namespace llvm::MachO;

struct WhyLiveEntry {
  InputSection *isec;
  // Keep track of the entry that caused us to mark `isec` as live.
  const WhyLiveEntry *prev;

  WhyLiveEntry(InputSection *isec, const WhyLiveEntry *prev)
      : isec(isec), prev(prev) {}
};

// Type-erased interface to MarkLiveImpl. Used for adding roots to the liveness
// graph.
class MarkLive {
public:
  virtual void enqueue(InputSection *isec, uint64_t off) = 0;
  virtual void addSym(Symbol *s) = 0;
  virtual void markTransitively() = 0;
  virtual ~MarkLive() = default;
};

// Sets `flag` and returns true if no other thread set it first.
static bool tryClaim(bool &flag) {
  std::atomic<bool> &atomicFlag = utils::atomicRef(flag);
  // Most claims find the flag already set, and the load lets those skip the
  // more expensive atomic exchange.
  return !atomicFlag.load(std::memory_order_relaxed) &&
         !atomicFlag.exchange(true, std::memory_order_relaxed);
}

// Marks liveness on as many threads as are available. -why_live output is
// deterministic only with --threads=1.
template <bool RecordWhyLive> class MarkLiveImpl : public MarkLive {
public:
  // -why_live is a rarely used option, so we don't want support for that flag
  // to slow down the main -dead_strip code path. As such, we employ templates
  // to avoid the usage of WhyLiveEntry in the main code path. This saves us
  // from needless allocations and pointer indirections.
  using WorklistEntry =
      std::conditional_t<RecordWhyLive, WhyLiveEntry, InputSection>;
  using Worklist = SmallVector<WorklistEntry *, 0>;

  void enqueue(InputSection *isec, uint64_t off) override {
    enqueue(isec, off, nullptr, nextLevel);
  }
  void addSym(Symbol *s) override { addSym(s, nullptr, nextLevel); }
  void markTransitively() override;

private:
  // Several threads call these at once, each passing its own worklist to
  // receive the sections it claims.
  void enqueue(InputSection *isec, uint64_t off, const WorklistEntry *prev,
               Worklist &worklist) const;
  // Most calls find `s` already claimed, so keep that check out from behind a
  // call.
  LLVM_ATTRIBUTE_ALWAYS_INLINE void addSym(Symbol *s, const WorklistEntry *prev,
                                           Worklist &worklist) const;
  void visitSymbol(Symbol *s, const WorklistEntry *prev,
                   Worklist &worklist) const;
  void visit(WorklistEntry *entry, Worklist &worklist) const;
  void drain();
  const InputSection *getInputSection(const WorklistEntry *) const;
  WorklistEntry *makeEntry(InputSection *, const WorklistEntry *prev) const;

  // Sections that have been marked live and that the next level of drain() will
  // visit. Literal sections cannot contain references to other sections, so we
  // only store ConcatInputSections here.
  Worklist nextLevel;
};

// Marks `isec` (at `off`) live. Only the thread whose claim succeeds queues a
// section.
template <bool RecordWhyLive>
void MarkLiveImpl<RecordWhyLive>::enqueue(InputSection *isec, uint64_t off,
                                          const WorklistEntry *prev,
                                          Worklist &worklist) const {
  if (auto *s = dyn_cast<ConcatInputSection>(isec)) {
    if (!tryClaim(s->live))
      return;
    assert(!s->isCoalescedWeak());
    worklist.push_back(makeEntry(s, prev));
  } else {
    isec->markLive(off);
  }
}

static void printWhyLive(const Symbol *s, const WhyLiveEntry *prev) {
  std::string out = toString(*s) + " from " + toString(s->getFile());
  int indent = 2;
  for (const WhyLiveEntry *entry = prev; entry;
       entry = entry->prev, indent += 2) {
    const TinyPtrVector<Defined *> &symbols = entry->isec->symbols;
    // With .subsections_with_symbols set, most isecs will have exactly one
    // entry in their symbols vector, so we just print the first one.
    if (!symbols.empty())
      out += "\n" + std::string(indent, ' ') + toString(*symbols.front()) +
             " from " + toString(symbols.front()->getFile());
  }
  message(out);
}

template <bool RecordWhyLive>
void MarkLiveImpl<RecordWhyLive>::addSym(Symbol *s, const WorklistEntry *prev,
                                         Worklist &worklist) const {
  if (tryClaim(s->used))
    visitSymbol(s, prev, worklist);
}

// Marks what `s` keeps alive. Called by the thread that claimed `s`. This is
// separate from addSym() because the class-stub edge makes it recursive, and
// GCC won't always-inline a function that calls itself.
template <bool RecordWhyLive>
void MarkLiveImpl<RecordWhyLive>::visitSymbol(Symbol *s,
                                              const WorklistEntry *prev,
                                              Worklist &worklist) const {
  if constexpr (RecordWhyLive)
    if (!config->whyLive.empty() && config->whyLive.match(s->getName()))
      printWhyLive(s, prev);
  if (auto *classSym = in.objcStubs->lookupClassSymbol(s))
    addSym(classSym, prev, worklist);
  if (auto *d = dyn_cast<Defined>(s)) {
    if (d->isec())
      enqueue(d->isec(), d->value, prev, worklist);
    if (d->unwindEntry())
      enqueue(d->unwindEntry(), 0, prev, worklist);
  }
}

template <bool RecordWhyLive>
void MarkLiveImpl<RecordWhyLive>::visit(WorklistEntry *entry,
                                        Worklist &worklist) const {
  // Entries that get placed onto the worklist always contain
  // ConcatInputSections. `WhyLiveEntry::prev` may point to entries that
  // contain other types of InputSections (due to S_ATTR_LIVE_SUPPORT), but
  // those entries should never be pushed onto the worklist.
  auto *isec = cast<ConcatInputSection>(getInputSection(entry));
  assert(isec->isLive(0) && "We mark as live when pushing onto the worklist!");

  // Mark all symbols listed in the relocation table for this section.
  for (const Relocation &r : isec->relocs) {
    if (auto *s = dyn_cast<Symbol *>(r.referent))
      addSym(s, entry, worklist);
    else
      enqueue(cast<InputSection *>(r.referent), r.getAddend(), entry, worklist);
  }
  for (Defined *d : isec->symbols)
    addSym(d, entry, worklist);
}

// Visits the sections in nextLevel, one level at a time. Workers take chunks of
// a level, small enough that every worker gets some, and visit them depth-first
// to take advantage of memory locality. Once one worker has run out of chunks,
// the others move the rest of their stacks to nextLevel if they have already
// visited a minimum number of sections. Thus the level quickly ends and all
// workers share the rest in the next one. With one thread, this reduces to a
// single depth-first walk. The last join makes every live bit visible to the
// live-support scan, which reads them with plain loads.
template <bool RecordWhyLive> void MarkLiveImpl<RecordWhyLive>::drain() {
  const size_t numThreads = parallel::getThreadCount();
  // The minimum number of visits a worker makes in a level before it may hand
  // back its stack. This keeps levels with fewer sections than workers from
  // ending after a visit or two, which would result in a lot of fork-join
  // overhead.
  constexpr size_t minVisitsPerLevel = 256;
  while (!nextLevel.empty()) {
    Worklist level = std::move(nextLevel);
    nextLevel.clear();
    std::mutex nextLevelMutex;
    std::atomic<bool> anyWorkerIdle{false};
    size_t chunkSize =
        std::clamp<size_t>(level.size() / (4 * numThreads), 1, 64);
    // Each worker starts with the chunk at its own index and then claims the
    // chunks after the first numThreads.
    std::atomic<size_t> next{numThreads * chunkSize};
    parallelFor(0, numThreads, [&](size_t shard) {
      Worklist stack;
      size_t visited = 0;
      for (size_t begin = shard * chunkSize; begin < level.size();
           begin = next.fetch_add(chunkSize, std::memory_order_relaxed)) {
        size_t end = std::min(begin + chunkSize, level.size());
        stack.append(level.begin() + begin, level.begin() + end);
        while (!stack.empty()) {
          visit(stack.pop_back_val(), stack);
          if (++visited >= minVisitsPerLevel && !stack.empty() &&
              anyWorkerIdle.load(std::memory_order_relaxed)) {
            std::lock_guard<std::mutex> lock(nextLevelMutex);
            nextLevel.append(stack.begin(), stack.end());
            stack.clear();
          }
        }
      }
      anyWorkerIdle.store(true, std::memory_order_relaxed);
    });
  }
}

template <bool RecordWhyLive>
const InputSection *MarkLiveImpl<RecordWhyLive>::getInputSection(
    const MarkLiveImpl<RecordWhyLive>::WorklistEntry *entry) const {
  if constexpr (RecordWhyLive)
    return entry->isec;
  else
    return entry;
}

template <bool RecordWhyLive>
typename MarkLiveImpl<RecordWhyLive>::WorklistEntry *
MarkLiveImpl<RecordWhyLive>::makeEntry(
    InputSection *isec,
    const MarkLiveImpl<RecordWhyLive>::WorklistEntry *prev) const {
  if constexpr (RecordWhyLive) {
    if (!isec) {
      assert(!prev);
      return nullptr;
    }
    return makeThreadLocal<WhyLiveEntry>(isec, prev);
  } else {
    return isec;
  }
}

template <bool RecordWhyLive>
void MarkLiveImpl<RecordWhyLive>::markTransitively() {
  do {
    // Mark things reachable from GC roots as live.
    drain();

    // S_ATTR_LIVE_SUPPORT sections are live if they point _to_ a live
    // section. Process them in a second pass.
    for (ConcatInputSection *isec : inputSections) {
      // FIXME: Check if copying all S_ATTR_LIVE_SUPPORT sections into a
      // separate vector and only walking that here is faster.
      if (!(isec->getFlags() & S_ATTR_LIVE_SUPPORT) || isec->live)
        continue;

      for (const Relocation &r : isec->relocs) {
        if (auto *s = dyn_cast<Symbol *>(r.referent)) {
          if (s->isLive()) {
            InputSection *referentIsec = nullptr;
            if (auto *d = dyn_cast<Defined>(s))
              referentIsec = d->isec();
            enqueue(isec, 0, makeEntry(referentIsec, nullptr), nextLevel);
          }
        } else {
          auto *referentIsec = cast<InputSection *>(r.referent);
          if (referentIsec->isLive(r.getAddend()))
            enqueue(isec, 0, makeEntry(referentIsec, nullptr), nextLevel);
        }
      }
    }

    // S_ATTR_LIVE_SUPPORT could have marked additional sections live,
    // which in turn could mark additional S_ATTR_LIVE_SUPPORT sections live.
    // Iterate. In practice, the second iteration won't mark additional
    // S_ATTR_LIVE_SUPPORT sections live.
  } while (!nextLevel.empty());
}

// Set live bit on for each reachable chunk. Unmarked (unreachable)
// InputSections will be ignored by Writer, so they will be excluded
// from the final output.
void markLive() {
  TimeTraceScope timeScope("markLive");
  MarkLive *marker;
  if (config->whyLive.empty())
    marker = make<MarkLiveImpl<false>>();
  else
    marker = make<MarkLiveImpl<true>>();
  // Add GC roots.
  if (config->entry)
    marker->addSym(config->entry);
  for (Symbol *sym : symtab->getSymbols()) {
    if (auto *defined = dyn_cast<Defined>(sym)) {
      // -exported_symbol(s_list)
      if (!config->exportedSymbols.empty() &&
          config->exportedSymbols.match(defined->getName())) {
        // NOTE: Even though exporting private externs is an ill-defined
        // operation, we are purposely not checking for privateExtern in
        // order to follow ld64's behavior of treating all exported private
        // extern symbols as live, irrespective of whether they are autohide.
        marker->addSym(defined);
        continue;
      }

      // public symbols explicitly marked .no_dead_strip
      if (defined->referencedDynamically || defined->noDeadStrip) {
        marker->addSym(defined);
        continue;
      }

      // FIXME: When we implement these flags, make symbols from them GC
      // roots:
      // * -reexported_symbol(s_list)
      // * -alias_list
      // * -init

      // In dylibs and bundles and in executables with -export_dynamic,
      // all external functions are GC roots.
      bool externsAreRoots =
          config->outputType != MH_EXECUTE || config->exportDynamic;
      if (externsAreRoots && !defined->privateExtern) {
        marker->addSym(defined);
        continue;
      }
    }
  }
  // -u symbols
  for (Symbol *sym : config->explicitUndefineds)
    marker->addSym(sym);
  // local symbols explicitly marked .no_dead_strip
  for (const InputFile *file : inputFiles)
    if (auto *objFile = dyn_cast<ObjFile>(file))
      for (Symbol *sym : objFile->symbols)
        if (auto *defined = dyn_cast_or_null<Defined>(sym))
          if (!defined->isExternal() && defined->noDeadStrip)
            marker->addSym(defined);
  if (auto *stubBinder =
          dyn_cast_or_null<DylibSymbol>(symtab->find("dyld_stub_binder")))
    marker->addSym(stubBinder);
  for (ConcatInputSection *isec : inputSections) {
    // Sections marked no_dead_strip
    if (isec->getFlags() & S_ATTR_NO_DEAD_STRIP) {
      marker->enqueue(isec, 0);
      continue;
    }

    // mod_init_funcs, mod_term_funcs sections
    if (sectionType(isec->getFlags()) == S_MOD_INIT_FUNC_POINTERS ||
        sectionType(isec->getFlags()) == S_MOD_TERM_FUNC_POINTERS) {
      assert(!config->emitInitOffsets ||
             sectionType(isec->getFlags()) != S_MOD_INIT_FUNC_POINTERS);
      marker->enqueue(isec, 0);
      continue;
    }
  }

  for (ConcatInputSection *isec : in.initOffsets->inputs())
    marker->enqueue(isec, 0);

  marker->markTransitively();
}

} // namespace lld::macho
