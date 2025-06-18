//===- MarkLive.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements --gc-sections, which is a feature to remove unused
// sections from output. Unused sections are sections that are not reachable
// from known GC-root symbols or sections. Naturally the feature is
// implemented as a mark-sweep garbage collector.
//
// Here's how it works. Each InputSectionBase has a "Live" bit. The bit is off
// by default. Starting with GC-root symbols or sections, markLive function
// defined in this file visits all reachable sections to set their Live
// bits. Writer will then ignore sections whose Live bits are off, so that
// such sections are not included into output.
//
//===----------------------------------------------------------------------===//

#include "MarkLive.h"
#include "InputFiles.h"
#include "InputSection.h"
#include "LinkerScript.h"
#include "SymbolTable.h"
#include "Symbols.h"
#include "SyntheticSections.h"
#include "Target.h"
#include "lld/Common/Strings.h"
#include "llvm/ADT/DenseMapInfoVariant.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TimeProfiler.h"
#include <variant>
#include <vector>

using namespace llvm;
using namespace llvm::ELF;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace lld;
using namespace lld::elf;

namespace {
using SecOffset = std::pair<InputSectionBase *, unsigned>;

// Something that can have an independent reason for being live.
using LiveItem = std::variant<InputSectionBase *, Symbol *, SecOffset>;

// The most proximate reason that something is live.
struct LiveReason {
  std::optional<LiveItem> item;
  StringRef desc;
};

template <class ELFT, bool TrackWhyLive> class MarkLive {
public:
  MarkLive(Ctx &ctx, unsigned partition) : ctx(ctx), partition(partition) {}

  void run();
  void moveToMain();
  void printWhyLive(Symbol *s) const;

private:
  void enqueue(InputSectionBase *sec, uint64_t offset, Symbol *sym,
               LiveReason reason);
  void markSymbol(Symbol *sym, StringRef reason);
  void mark();

  template <class RelTy>
  void resolveReloc(InputSectionBase &sec, RelTy &rel, bool fromFDE);

  template <class RelTy>
  void scanEhFrameSection(EhInputSection &eh, ArrayRef<RelTy> rels);

  Ctx &ctx;
  // The index of the partition that we are currently processing.
  unsigned partition;

  // A list of sections to visit.
  SmallVector<InputSection *, 0> queue;

  // There are normally few input sections whose names are valid C
  // identifiers, so we just store a SmallVector instead of a multimap.
  DenseMap<StringRef, SmallVector<InputSectionBase *, 0>> cNamedSections;

  // The most proximate reason that something is live. This forms a DAG between
  // LiveItems. Acyclicality is maintained by only admitting the first
  // discovered reason for each LiveItem; this captures the acyclic region of
  // the liveness graph around the GC roots.
  DenseMap<LiveItem, LiveReason> whyLive;
};
} // namespace

template <class ELFT>
static uint64_t getAddend(Ctx &ctx, InputSectionBase &sec,
                          const typename ELFT::Rel &rel) {
  return ctx.target->getImplicitAddend(sec.content().begin() + rel.r_offset,
                                       rel.getType(ctx.arg.isMips64EL));
}

template <class ELFT>
static uint64_t getAddend(Ctx &, InputSectionBase &sec,
                          const typename ELFT::Rela &rel) {
  return rel.r_addend;
}

// Currently, we assume all input CREL relocations have an explicit addend.
template <class ELFT>
static uint64_t getAddend(Ctx &, InputSectionBase &sec,
                          const typename ELFT::Crel &rel) {
  return rel.r_addend;
}

template <class ELFT, bool TrackWhyLive>
template <class RelTy>
void MarkLive<ELFT, TrackWhyLive>::resolveReloc(InputSectionBase &sec,
                                                RelTy &rel, bool fromFDE) {
  // If a symbol is referenced in a live section, it is used.
  Symbol &sym = sec.file->getRelocTargetSym(rel);
  sym.used = true;

  LiveReason reason;
  if (TrackWhyLive)
    reason = {SecOffset(&sec, rel.r_offset), "referenced by"};

  if (auto *d = dyn_cast<Defined>(&sym)) {
    auto *relSec = dyn_cast_or_null<InputSectionBase>(d->section);
    if (!relSec)
      return;

    uint64_t offset = d->value;
    if (d->isSection())
      offset += getAddend<ELFT>(ctx, sec, rel);

    // fromFDE being true means this is referenced by a FDE in a .eh_frame
    // piece. The relocation points to the described function or to a LSDA. We
    // only need to keep the LSDA live, so ignore anything that points to
    // executable sections. If the LSDA is in a section group or has the
    // SHF_LINK_ORDER flag, we ignore the relocation as well because (a) if the
    // associated text section is live, the LSDA will be retained due to section
    // group/SHF_LINK_ORDER rules (b) if the associated text section should be
    // discarded, marking the LSDA will unnecessarily retain the text section.
    if (!(fromFDE && ((relSec->flags & (SHF_EXECINSTR | SHF_LINK_ORDER)) ||
                      relSec->nextInSectionGroup))) {
      Symbol *canonicalSym = d;
      if (TrackWhyLive && d->isSection()) {
        // This is expensive, so ideally this would be deferred until it's known
        // whether this reference contributes to a printed whyLive chain, but
        // that determination cannot be made without knowing the enclosing
        // symbol.
        if (Symbol *s = relSec->getEnclosingSymbol(offset))
          canonicalSym = s;
        else
          canonicalSym = nullptr;
      }
      enqueue(relSec, offset, canonicalSym, reason);
    }
    return;
  }

  if (auto *ss = dyn_cast<SharedSymbol>(&sym)) {
    if (!ss->isWeak()) {
      cast<SharedFile>(ss->file)->isNeeded = true;
      if (TrackWhyLive)
        whyLive.try_emplace(&sym, reason);
    }
  }

  for (InputSectionBase *sec : cNamedSections.lookup(sym.getName()))
    enqueue(sec, /*offset=*/0, /*sym=*/nullptr, reason);
}

// The .eh_frame section is an unfortunate special case.
// The section is divided in CIEs and FDEs and the relocations it can have are
// * CIEs can refer to a personality function.
// * FDEs can refer to a LSDA
// * FDEs refer to the function they contain information about
// The last kind of relocation cannot keep the referred section alive, or they
// would keep everything alive in a common object file. In fact, each FDE is
// alive if the section it refers to is alive.
// To keep things simple, in here we just ignore the last relocation kind. The
// other two keep the referred section alive.
//
// A possible improvement would be to fully process .eh_frame in the middle of
// the gc pass. With that we would be able to also gc some sections holding
// LSDAs and personality functions if we found that they were unused.
template <class ELFT, bool TrackWhyLive>
template <class RelTy>
void MarkLive<ELFT, TrackWhyLive>::scanEhFrameSection(EhInputSection &eh,
                                                      ArrayRef<RelTy> rels) {
  for (const EhSectionPiece &cie : eh.cies)
    if (cie.firstRelocation != unsigned(-1))
      resolveReloc(eh, rels[cie.firstRelocation], false);
  for (const EhSectionPiece &fde : eh.fdes) {
    size_t firstRelI = fde.firstRelocation;
    if (firstRelI == (unsigned)-1)
      continue;
    uint64_t pieceEnd = fde.inputOff + fde.size;
    for (size_t j = firstRelI, end2 = rels.size();
         j < end2 && rels[j].r_offset < pieceEnd; ++j)
      resolveReloc(eh, rels[j], true);
  }
}

// Some sections are used directly by the loader, so they should never be
// garbage-collected. This function returns true if a given section is such
// section.
static bool isReserved(InputSectionBase *sec) {
  switch (sec->type) {
  case SHT_FINI_ARRAY:
  case SHT_INIT_ARRAY:
  case SHT_PREINIT_ARRAY:
    return true;
  case SHT_NOTE:
    // SHT_NOTE sections in a group are subject to garbage collection.
    return !sec->nextInSectionGroup;
  default:
    // Support SHT_PROGBITS .init_array (https://golang.org/issue/50295) and
    // .init_array.N (https://github.com/rust-lang/rust/issues/92181) for a
    // while.
    StringRef s = sec->name;
    return s == ".init" || s == ".fini" || s.starts_with(".init_array") ||
           s == ".jcr" || s.starts_with(".ctors") || s.starts_with(".dtors");
  }
}

template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::enqueue(InputSectionBase *sec,
                                           uint64_t offset, Symbol *sym,
                                           LiveReason reason) {
  // Usually, a whole section is marked as live or dead, but in mergeable
  // (splittable) sections, each piece of data has independent liveness bit.
  // So we explicitly tell it which offset is in use.
  if (auto *ms = dyn_cast<MergeInputSection>(sec))
    ms->getSectionPiece(offset).live = true;

  // Set Sec->Partition to the meet (i.e. the "minimum") of Partition and
  // Sec->Partition in the following lattice: 1 < other < 0. If Sec->Partition
  // doesn't change, we don't need to do anything.
  if (sec->partition == 1 || sec->partition == partition)
    return;
  sec->partition = sec->partition ? 1 : partition;

  if (TrackWhyLive) {
    if (sym) {
      // If a specific symbol is referenced, that keeps it live. The symbol then
      // keeps its section live.
      whyLive.try_emplace(sym, reason);
      whyLive.try_emplace(sec, LiveReason{sym, "contained live symbol"});
    } else {
      // Otherwise, the reference generically keeps the section live.
      whyLive.try_emplace(sec, reason);
    }
  }

  // Add input section to the queue.
  if (InputSection *s = dyn_cast<InputSection>(sec))
    queue.push_back(s);
}

// Print the stack of reasons that the given symbol is live.
template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::printWhyLive(Symbol *s) const {
  // Skip dead symbols. A symbol is dead if it belongs to a dead section.
  if (auto *d = dyn_cast<Defined>(s)) {
    auto *sec = dyn_cast_or_null<InputSectionBase>(d->section);
    if (sec && !sec->isLive())
      return;
  }

  auto msg = Msg(ctx);

  const auto printSymbol = [&](Symbol *s) {
    msg << s->file << ":(" << s << ')';
  };

  msg << "live symbol: ";
  printSymbol(s);

  LiveItem cur = s;
  while (true) {
    auto it = whyLive.find(cur);
    LiveReason reason;
    // If there is a specific reason this item is live...
    if (it != whyLive.end()) {
      reason = it->second;
    } else {
      // This item is live, but it has no tracked reason. It must be an
      // unreferenced symbol in a live section or a symbol with no section.
      InputSectionBase *sec = nullptr;
      if (auto *d = dyn_cast<Defined>(std::get<Symbol *>(cur)))
        sec = dyn_cast_or_null<InputSectionBase>(d->section);
      reason = sec ? LiveReason{sec, "in live section"}
                   : LiveReason{std::nullopt, "no section"};
    }

    if (!reason.item) {
      msg << " (" << reason.desc << ')';
      break;
    }

    msg << "\n>>> " << reason.desc << ": ";
    // The reason may not yet have been resolved to a symbol; do so now.
    if (std::holds_alternative<SecOffset>(*reason.item)) {
      const auto &so = std::get<SecOffset>(*reason.item);
      InputSectionBase *sec = so.first;
      Defined *sym = sec->getEnclosingSymbol(so.second);
      cur = sym ? LiveItem(sym) : LiveItem(sec);
    } else {
      cur = *reason.item;
    }

    if (std::holds_alternative<Symbol *>(cur))
      printSymbol(std::get<Symbol *>(cur));
    else
      msg << std::get<InputSectionBase *>(cur);
  }
}

template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::markSymbol(Symbol *sym, StringRef reason) {
  if (auto *d = dyn_cast_or_null<Defined>(sym))
    if (auto *isec = dyn_cast_or_null<InputSectionBase>(d->section))
      enqueue(isec, d->value, sym, {std::nullopt, reason});
}

// This is the main function of the garbage collector.
// Starting from GC-root sections, this function visits all reachable
// sections to set their "Live" bits.
template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::run() {
  // Add GC root symbols.

  // Preserve externally-visible symbols if the symbols defined by this
  // file can interpose other ELF file's symbols at runtime.
  for (Symbol *sym : ctx.symtab->getSymbols())
    if (sym->isExported && sym->partition == partition)
      markSymbol(sym, "externally visible symbol; may interpose");

  // If this isn't the main partition, that's all that we need to preserve.
  if (partition != 1) {
    mark();
    return;
  }

  markSymbol(ctx.symtab->find(ctx.arg.entry), "entry point");
  markSymbol(ctx.symtab->find(ctx.arg.init), "initializer function");
  markSymbol(ctx.symtab->find(ctx.arg.fini), "finalizer function");
  for (StringRef s : ctx.arg.undefined)
    markSymbol(ctx.symtab->find(s), "undefined command line flag");
  for (StringRef s : ctx.script->referencedSymbols)
    markSymbol(ctx.symtab->find(s), "referenced by linker script");
  for (auto [symName, _] : ctx.symtab->cmseSymMap) {
    markSymbol(ctx.symtab->cmseSymMap[symName].sym, "ARM CMSE symbol");
    markSymbol(ctx.symtab->cmseSymMap[symName].acleSeSym, "ARM CMSE symbol");
  }

  // Mark .eh_frame sections as live because there are usually no relocations
  // that point to .eh_frames. Otherwise, the garbage collector would drop
  // all of them. We also want to preserve personality routines and LSDA
  // referenced by .eh_frame sections, so we scan them for that here.
  for (EhInputSection *eh : ctx.ehInputSections) {
    const RelsOrRelas<ELFT> rels =
        eh->template relsOrRelas<ELFT>(/*supportsCrel=*/false);
    if (rels.areRelocsRel())
      scanEhFrameSection(*eh, rels.rels);
    else if (rels.relas.size())
      scanEhFrameSection(*eh, rels.relas);
  }
  for (InputSectionBase *sec : ctx.inputSections) {
    if (sec->flags & SHF_GNU_RETAIN) {
      enqueue(sec, /*offset=*/0, /*sym=*/nullptr, {std::nullopt, "retained"});
      continue;
    }
    if (sec->flags & SHF_LINK_ORDER)
      continue;

    // Usually, non-SHF_ALLOC sections are not removed even if they are
    // unreachable through relocations because reachability is not a good signal
    // whether they are garbage or not (e.g. there is usually no section
    // referring to a .comment section, but we want to keep it.) When a
    // non-SHF_ALLOC section is retained, we also retain sections dependent on
    // it.
    //
    // Note on SHF_LINK_ORDER: Such sections contain metadata and they
    // have a reverse dependency on the InputSection they are linked with.
    // We are able to garbage collect them.
    //
    // Note on SHF_REL{,A}: Such sections reach here only when -r
    // or --emit-reloc were given. And they are subject of garbage
    // collection because, if we remove a text section, we also
    // remove its relocation section.
    //
    // Note on nextInSectionGroup: The ELF spec says that group sections are
    // included or omitted as a unit. We take the interpretation that:
    //
    // - Group members (nextInSectionGroup != nullptr) are subject to garbage
    //   collection.
    // - Groups members are retained or discarded as a unit.
    if (!(sec->flags & SHF_ALLOC)) {
      if (!isStaticRelSecType(sec->type) && !sec->nextInSectionGroup) {
        sec->markLive();
        for (InputSection *isec : sec->dependentSections)
          isec->markLive();
      }
    }

    // Preserve special sections and those which are specified in linker
    // script KEEP command.
    if (isReserved(sec)) {
      enqueue(sec, /*offset=*/0, /*sym=*/nullptr, {std::nullopt, "reserved"});
    } else if (ctx.script->shouldKeep(sec)) {
      enqueue(sec, /*offset=*/0, /*sym=*/nullptr,
              {std::nullopt, "KEEP in linker script"});
    } else if ((!ctx.arg.zStartStopGC || sec->name.starts_with("__libc_")) &&
               isValidCIdentifier(sec->name)) {
      // As a workaround for glibc libc.a before 2.34
      // (https://sourceware.org/PR27492), retain __libc_atexit and similar
      // sections regardless of zStartStopGC.
      cNamedSections[ctx.saver.save("__start_" + sec->name)].push_back(sec);
      cNamedSections[ctx.saver.save("__stop_" + sec->name)].push_back(sec);
    }
  }

  mark();

  if (TrackWhyLive) {
    const auto handleSym = [&](Symbol *sym) {
      if (llvm::any_of(ctx.arg.whyLive, [sym](const llvm::GlobPattern &pat) {
            return pat.match(sym->getName());
          }))
        printWhyLive(sym);
    };

    for (Symbol *sym : ctx.symtab->getSymbols())
      handleSym(sym);
    for (ELFFileBase *file : ctx.objectFiles)
      for (Symbol *sym : file->getSymbols())
        if (sym->isLocal())
          handleSym(sym);
  }
}

template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::mark() {
  // Mark all reachable sections.
  while (!queue.empty()) {
    InputSectionBase &sec = *queue.pop_back_val();

    const RelsOrRelas<ELFT> rels = sec.template relsOrRelas<ELFT>();
    for (const typename ELFT::Rel &rel : rels.rels)
      resolveReloc(sec, rel, false);
    for (const typename ELFT::Rela &rel : rels.relas)
      resolveReloc(sec, rel, false);
    for (const typename ELFT::Crel &rel : rels.crels)
      resolveReloc(sec, rel, false);

    for (InputSectionBase *isec : sec.dependentSections)
      enqueue(isec, /*offset=*/0, /*sym=*/nullptr,
              {&sec, "depended on by section"});

    // Mark the next group member.
    if (sec.nextInSectionGroup)
      enqueue(sec.nextInSectionGroup, /*offset=*/0, /*sym=*/nullptr,
              {&sec, "in section group with"});
  }
}

// Move the sections for some symbols to the main partition, specifically ifuncs
// (because they can result in an IRELATIVE being added to the main partition's
// GOT, which means that the ifunc must be available when the main partition is
// loaded) and TLS symbols (because we only know how to correctly process TLS
// relocations for the main partition).
//
// We also need to move sections whose names are C identifiers that are referred
// to from __start_/__stop_ symbols because there will only be one set of
// symbols for the whole program.
template <class ELFT, bool TrackWhyLive>
void MarkLive<ELFT, TrackWhyLive>::moveToMain() {
  for (ELFFileBase *file : ctx.objectFiles)
    for (Symbol *s : file->getSymbols())
      if (auto *d = dyn_cast<Defined>(s))
        if ((d->type == STT_GNU_IFUNC || d->type == STT_TLS) && d->section &&
            d->section->isLive())
          markSymbol(s, /*reason=*/{});

  for (InputSectionBase *sec : ctx.inputSections) {
    if (!sec->isLive() || !isValidCIdentifier(sec->name))
      continue;
    if (ctx.symtab->find(("__start_" + sec->name).str()) ||
        ctx.symtab->find(("__stop_" + sec->name).str()))
      enqueue(sec, /*offset=*/0, /*sym=*/nullptr, /*reason=*/{});
  }

  mark();
}

// Before calling this function, Live bits are off for all
// input sections. This function make some or all of them on
// so that they are emitted to the output file.
template <class ELFT> void elf::markLive(Ctx &ctx) {
  llvm::TimeTraceScope timeScope("markLive");
  // If --gc-sections is not given, retain all input sections.
  if (!ctx.arg.gcSections) {
    // If a DSO defines a symbol referenced in a regular object, it is needed.
    for (Symbol *sym : ctx.symtab->getSymbols())
      if (auto *s = dyn_cast<SharedSymbol>(sym))
        if (s->isUsedInRegularObj && !s->isWeak())
          cast<SharedFile>(s->file)->isNeeded = true;
    return;
  }

  for (InputSectionBase *sec : ctx.inputSections)
    sec->markDead();

  // Follow the graph to mark all live sections.
  for (unsigned i = 1, e = ctx.partitions.size(); i <= e; ++i)
    if (ctx.arg.whyLive.empty())
      MarkLive<ELFT, false>(ctx, i).run();
    else
      MarkLive<ELFT, true>(ctx, i).run();

  // If we have multiple partitions, some sections need to live in the main
  // partition even if they were allocated to a loadable partition. Move them
  // there now.
  if (ctx.partitions.size() != 1)
    MarkLive<ELFT, false>(ctx, 1).moveToMain();

  // Report garbage-collected sections.
  if (ctx.arg.printGcSections)
    for (InputSectionBase *sec : ctx.inputSections)
      if (!sec->isLive())
        Msg(ctx) << "removing unused section " << sec;
}

template void elf::markLive<ELF32LE>(Ctx &);
template void elf::markLive<ELF32BE>(Ctx &);
template void elf::markLive<ELF64LE>(Ctx &);
template void elf::markLive<ELF64BE>(Ctx &);
