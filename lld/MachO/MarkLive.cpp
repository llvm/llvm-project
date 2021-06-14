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

// Set live bit on for each reachable chunk. Unmarked (unreachable)
// InputSections will be ignored by Writer, so they will be excluded
// from the final output.
void markLive() {
  TimeTraceScope timeScope("markLive");

  // We build up a worklist of sections which have been marked as live. We only
  // push into the worklist when we discover an unmarked section, and we mark
  // as we push, so sections never appear twice in the list.
  // Literal sections cannot contain references to other sections, so we only
  // store ConcatInputSections in our worklist.
  SmallVector<ConcatInputSection *, 256> worklist;

  auto enqueue = [&](InputSection *isec, uint64_t off) {
    if (isec->isLive(off))
      return;
    isec->markLive(off);
    if (auto s = dyn_cast<ConcatInputSection>(isec)) {
      assert(!s->isCoalescedWeak());
      worklist.push_back(s);
    }
  };

  auto addSym = [&](Symbol *s) {
    s->used = true;
    if (auto *d = dyn_cast<Defined>(s))
      if (d->isec)
        enqueue(d->isec, d->value);
  };

  // Add GC roots.
  if (config->entry)
    addSym(config->entry);
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
        addSym(defined);
        continue;
      }

      // public symbols explicitly marked .no_dead_strip
      if (defined->referencedDynamically || defined->noDeadStrip) {
        addSym(defined);
        continue;
      }

      // FIXME: When we implement these flags, make symbols from them GC roots:
      // * -reexported_symbol(s_list)
      // * -alias(-list)
      // * -init

      // In dylibs and bundles, all external functions are GC roots.
      // FIXME: -export_dynamic should enable this for executables too.
      if (config->outputType != MH_EXECUTE && !defined->privateExtern) {
        addSym(defined);
        continue;
      }
    }
  }
  // -u symbols
  for (Symbol *sym : config->explicitUndefineds)
    if (auto *defined = dyn_cast<Defined>(sym))
      addSym(defined);
  // local symbols explicitly marked .no_dead_strip
  for (const InputFile *file : inputFiles)
    if (auto *objFile = dyn_cast<ObjFile>(file))
      for (Symbol *sym : objFile->symbols)
        if (auto *defined = dyn_cast_or_null<Defined>(sym))
          if (!defined->isExternal() && defined->noDeadStrip)
            addSym(defined);
  if (auto *stubBinder =
          dyn_cast_or_null<DylibSymbol>(symtab->find("dyld_stub_binder")))
    addSym(stubBinder);
  for (InputSection *isec : inputSections) {
    // Sections marked no_dead_strip
    if (isec->flags & S_ATTR_NO_DEAD_STRIP) {
      assert(isa<ConcatInputSection>(isec));
      enqueue(isec, 0);
      continue;
    }

    // mod_init_funcs, mod_term_funcs sections
    if (sectionType(isec->flags) == S_MOD_INIT_FUNC_POINTERS ||
        sectionType(isec->flags) == S_MOD_TERM_FUNC_POINTERS) {
      assert(isa<ConcatInputSection>(isec));
      enqueue(isec, 0);
      continue;
    }

    // Dead strip runs before UnwindInfoSection handling so we need to keep
    // __LD,__compact_unwind alive here.
    // But that section contains absolute references to __TEXT,__text and
    // keeps most code alive due to that. So we can't just enqueue() the
    // section: We must skip the relocations for the functionAddress
    // in each CompactUnwindEntry.
    // See also scanEhFrameSection() in lld/ELF/MarkLive.cpp.
    if (isec->segname == segment_names::ld &&
        isec->name == section_names::compactUnwind) {
      auto concatIsec = cast<ConcatInputSection>(isec);
      concatIsec->live = true;
      const int compactUnwindEntrySize =
          target->wordSize == 8 ? sizeof(CompactUnwindEntry<uint64_t>)
                                : sizeof(CompactUnwindEntry<uint32_t>);
      for (const Reloc &r : isec->relocs) {
        // This is the relocation for the address of the function itself.
        // Ignore it, else these would keep everything alive.
        if (r.offset % compactUnwindEntrySize == 0)
          continue;

        if (auto *s = r.referent.dyn_cast<Symbol *>())
          addSym(s);
        else
          enqueue(r.referent.get<InputSection *>(), r.addend);
      }
      continue;
    }
  }

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
    }

    // S_ATTR_LIVE_SUPPORT sections are live if they point _to_ a live section.
    // Process them in a second pass.
    for (InputSection *isec : inputSections) {
      if (!isa<ConcatInputSection>(isec))
        continue;
      auto concatIsec = cast<ConcatInputSection>(isec);
      // FIXME: Check if copying all S_ATTR_LIVE_SUPPORT sections into a
      // separate vector and only walking that here is faster.
      if (!(concatIsec->flags & S_ATTR_LIVE_SUPPORT) || concatIsec->live)
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

} // namespace macho
} // namespace lld
