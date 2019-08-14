//===-- stats.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_STATS_H_
#define SCUDO_STATS_H_

#include "atomic_helpers.h"
#include "mutex.h"

#include <string.h>

namespace scudo {

// Memory allocator statistics
enum StatType { StatAllocated, StatFree, StatMapped, StatCount };

typedef uptr StatCounters[StatCount];

// Per-thread stats, live in per-thread cache. We use atomics so that the
// numbers themselves are consistent. But we don't use atomic_{add|sub} or a
// lock, because those are expensive operations , and we only care for the stats
// to be "somewhat" correct: eg. if we call GlobalStats::get while a thread is
// LocalStats::add'ing, this is OK, we will still get a meaningful number.
class LocalStats {
public:
  void initLinkerInitialized() {}
  void init() { memset(this, 0, sizeof(*this)); }

  void add(StatType I, uptr V) {
    V += atomic_load_relaxed(&StatsArray[I]);
    atomic_store_relaxed(&StatsArray[I], V);
  }

  void sub(StatType I, uptr V) {
    V = atomic_load_relaxed(&StatsArray[I]) - V;
    atomic_store_relaxed(&StatsArray[I], V);
  }

  void set(StatType I, uptr V) { atomic_store_relaxed(&StatsArray[I], V); }

  uptr get(StatType I) const { return atomic_load_relaxed(&StatsArray[I]); }

private:
  friend class GlobalStats;
  atomic_uptr StatsArray[StatCount];
  LocalStats *Next;
  LocalStats *Prev;
};

// Global stats, used for aggregation and querying.
class GlobalStats : public LocalStats {
public:
  void initLinkerInitialized() {
    Next = this;
    Prev = this;
  }
  void init() {
    memset(this, 0, sizeof(*this));
    initLinkerInitialized();
  }

  void link(LocalStats *S) {
    ScopedLock L(Mutex);
    S->Next = Next;
    S->Prev = this;
    Next->Prev = S;
    Next = S;
  }

  void unlink(LocalStats *S) {
    ScopedLock L(Mutex);
    S->Prev->Next = S->Next;
    S->Next->Prev = S->Prev;
    for (uptr I = 0; I < StatCount; I++)
      add(static_cast<StatType>(I), S->get(static_cast<StatType>(I)));
  }

  void get(uptr *S) const {
    memset(S, 0, StatCount * sizeof(uptr));
    ScopedLock L(Mutex);
    const LocalStats *Stats = this;
    for (;;) {
      for (uptr I = 0; I < StatCount; I++)
        S[I] += Stats->get(static_cast<StatType>(I));
      Stats = Stats->Next;
      if (Stats == this)
        break;
    }
    // All stats must be non-negative.
    for (uptr I = 0; I < StatCount; I++)
      S[I] = static_cast<sptr>(S[I]) >= 0 ? S[I] : 0;
  }

private:
  mutable HybridMutex Mutex;
};

} // namespace scudo

#endif // SCUDO_STATS_H_
