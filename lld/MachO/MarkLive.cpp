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
#include "UnwindInfoSection.h"
#include "mach-o/compact_unwind_encoding.h"
#include "llvm/Support/TimeProfiler.h"

namespace lld {
namespace macho {

using namespace llvm;
using namespace llvm::MachO;

class MarkLive {
public:
  void enqueue(InputSection *isec, uint64_t off);
  void addSym(Symbol *s);
  void markTransitively();

private:
  // We build up a worklist of sections which have been marked as live. We
  // only push into the worklist when we discover an unmarked section, and we
  // mark as we push, so sections never appear twice in the list. Literal
  // sections cannot contain references to other sections, so we only store
  // ConcatInputSections in our worklist.
  SmallVector<ConcatInputSection *, 256> worklist;
};

void MarkLive::enqueue(InputSection *isec, uint64_t off) {
  if (isec->isLive(off))
    return;
  isec->markLive(off);
  if (auto s = dyn_cast<ConcatInputSection>(isec)) {
    assert(!s->isCoalescedWeak());
    worklist.push_back(s);
  }
}

void MarkLive::addSym(Symbol *s) {
  if (s->used)
    return;
  s->used = true;
  if (auto *d = dyn_cast<Defined>(s)) {
    if (d->isec)
      enqueue(d->isec, d->value);
    if (d->unwindEntry)
      enqueue(d->unwindEntry, 0);
  }
}

void MarkLive::markTransitively() {
  do {
    // Mark things reachable from GC roots as live.
    while (!worklist.empty()) {
      ConcatInputSection *s = worklist.pop_back_val();
      assert(s->live && "We mark as live when pushing onto the worklist!");

      // Mark all symbols listed in the relocation table for this section.
      for (const Reloc &r : s->relocs) {
        if (auto *s = r.referent.dyn_cast<Symbol *>())
          addSym(s);
        else
          enqueue(r.referent.get<InputSection *>(), r.addend);
      }
      for (Defined *d : s->symbols)
        addSym(d);
    }

    // S_ATTR_LIVE_SUPPORT sections are live if they point _to_ a live
    // section. Process them in a second pass.
    for (ConcatInputSection *isec : inputSections) {
      // FIXME: Check if copying all S_ATTR_LIVE_SUPPORT sections into a
      // separate vector and only walking that here is faster.
      if (!(isec->getFlags() & S_ATTR_LIVE_SUPPORT) || isec->live)
        continue;

      for (const Reloc &r : isec->relocs) {
        bool referentLive;
        if (auto *s = r.referent.dyn_cast<Symbol *>())
          referentLive = s->isLive();
        else
          referentLive = r.referent.get<InputSection *>()->isLive(r.addend);
        if (referentLive)
          enqueue(isec, 0);
      }
    }

    // S_ATTR_LIVE_SUPPORT could have marked additional sections live,
    // which in turn could mark additional S_ATTR_LIVE_SUPPORT sections live.
    // Iterate. In practice, the second iteration won't mark additional
    // S_ATTR_LIVE_SUPPORT sections live.
  } while (!worklist.empty());
}

// Set live bit on for each reachable chunk. Unmarked (unreachable)
// InputSections will be ignored by Writer, so they will be excluded
// from the final output.
void markLive() {
  TimeTraceScope timeScope("markLive");
  MarkLive marker;
  // Add GC roots.
  if (config->entry)
    marker.addSym(config->entry);
  for (Symbol *sym : symtab->getSymbols()) {
    if (auto *defined = dyn_cast<Defined>(sym)) {
      // -exported_symbol(s_list)
      if (!config->exportedSymbols.empty() &&
          config->exportedSymbols.match(defined->getName())) {
        // FIXME: Instead of doing this here, maybe the Driver code doing
        // the matching should add them to explicitUndefineds? Then the
        // explicitUndefineds code below would handle this automatically.
        assert(!defined->privateExtern &&
               "should have been rejected by driver");
        marker.addSym(defined);
        continue;
      }

      // public symbols explicitly marked .no_dead_strip
      if (defined->referencedDynamically || defined->noDeadStrip) {
        marker.addSym(defined);
        continue;
      }

      // FIXME: When we implement these flags, make symbols from them GC
      // roots:
      // * -reexported_symbol(s_list)
      // * -alias(-list)
      // * -init

      // In dylibs and bundles and in executables with -export_dynamic,
      // all external functions are GC roots.
      bool externsAreRoots =
          config->outputType != MH_EXECUTE || config->exportDynamic;
      if (externsAreRoots && !defined->privateExtern) {
        marker.addSym(defined);
        continue;
      }
    }
  }
  // -u symbols
  for (Symbol *sym : config->explicitUndefineds)
    marker.addSym(sym);
  // local symbols explicitly marked .no_dead_strip
  for (const InputFile *file : inputFiles)
    if (auto *objFile = dyn_cast<ObjFile>(file))
      for (Symbol *sym : objFile->symbols)
        if (auto *defined = dyn_cast_or_null<Defined>(sym))
          if (!defined->isExternal() && defined->noDeadStrip)
            marker.addSym(defined);
  if (auto *stubBinder =
          dyn_cast_or_null<DylibSymbol>(symtab->find("dyld_stub_binder")))
    marker.addSym(stubBinder);
  for (ConcatInputSection *isec : inputSections) {
    // Sections marked no_dead_strip
    if (isec->getFlags() & S_ATTR_NO_DEAD_STRIP) {
      marker.enqueue(isec, 0);
      continue;
    }

    // mod_init_funcs, mod_term_funcs sections
    if (sectionType(isec->getFlags()) == S_MOD_INIT_FUNC_POINTERS ||
        sectionType(isec->getFlags()) == S_MOD_TERM_FUNC_POINTERS) {
      marker.enqueue(isec, 0);
      continue;
    }
  }

  marker.markTransitively();
}

} // namespace macho
} // namespace lld
