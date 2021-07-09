//===-- memprof_allocator.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemProfiler, a memory profiler.
//
// Implementation of MemProf's memory allocator, which uses the allocator
// from sanitizer_common.
//
//===----------------------------------------------------------------------===//

#include "memprof_allocator.h"
#include "memprof_mapping.h"
#include "memprof_stack.h"
#include "memprof_thread.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_file.h"
#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_list.h"
#include "sanitizer_common/sanitizer_stackdepot.h"

#include <sched.h>
#include <stdlib.h>
#include <time.h>

namespace __memprof {

static int GetCpuId(void) {
  // _memprof_preinit is called via the preinit_array, which subsequently calls
  // malloc. Since this is before _dl_init calls VDSO_SETUP, sched_getcpu
  // will seg fault as the address of __vdso_getcpu will be null.
  if (!memprof_init_done)
    return -1;
  return sched_getcpu();
}

// Compute the timestamp in ms.
static int GetTimestamp(void) {
  // timespec_get will segfault if called from dl_init
  if (!memprof_timestamp_inited) {
    // By returning 0, this will be effectively treated as being
    // timestamped at memprof init time (when memprof_init_timestamp_s
    // is initialized).
    return 0;
  }
  timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return (ts.tv_sec - memprof_init_timestamp_s) * 1000 + ts.tv_nsec / 1000000;
}

static MemprofAllocator &get_allocator();

// The memory chunk allocated from the underlying allocator looks like this:
// H H U U U U U U
//   H -- ChunkHeader (32 bytes)
//   U -- user memory.

// If there is left padding before the ChunkHeader (due to use of memalign),
// we store a magic value in the first uptr word of the memory block and
// store the address of ChunkHeader in the next uptr.
// M B L L L L L L L L L  H H U U U U U U
//   |                    ^
//   ---------------------|
//   M -- magic value kAllocBegMagic
//   B -- address of ChunkHeader pointing to the first 'H'

constexpr uptr kMaxAllowedMallocBits = 40;

// Should be no more than 32-bytes
struct ChunkHeader {
  // 1-st 4 bytes.
  u32 alloc_context_id;
  // 2-nd 4 bytes
  u32 cpu_id;
  // 3-rd 4 bytes
  u32 timestamp_ms;
  // 4-th 4 bytes
  // Note only 1 bit is needed for this flag if we need space in the future for
  // more fields.
  u32 from_memalign;
  // 5-th and 6-th 4 bytes
  // The max size of an allocation is 2^40 (kMaxAllowedMallocSize), so this
  // could be shrunk to kMaxAllowedMallocBits if we need space in the future for
  // more fields.
  atomic_uint64_t user_requested_size;
  // 23 bits available
  // 7-th and 8-th 4 bytes
  u64 data_type_id; // TODO: hash of type name
};

static const uptr kChunkHeaderSize = sizeof(ChunkHeader);
COMPILER_CHECK(kChunkHeaderSize == 32);

struct MemprofChunk : ChunkHeader {
  uptr Beg() { return reinterpret_cast<uptr>(this) + kChunkHeaderSize; }
  uptr UsedSize() {
    return atomic_load(&user_requested_size, memory_order_relaxed);
  }
  void *AllocBeg() {
    if (from_memalign)
      return get_allocator().GetBlockBegin(reinterpret_cast<void *>(this));
    return reinterpret_cast<void *>(this);
  }
};

class LargeChunkHeader {
  static constexpr uptr kAllocBegMagic =
      FIRST_32_SECOND_64(0xCC6E96B9, 0xCC6E96B9CC6E96B9ULL);
  atomic_uintptr_t magic;
  MemprofChunk *chunk_header;

public:
  MemprofChunk *Get() const {
    return atomic_load(&magic, memory_order_acquire) == kAllocBegMagic
               ? chunk_header
               : nullptr;
  }

  void Set(MemprofChunk *p) {
    if (p) {
      chunk_header = p;
      atomic_store(&magic, kAllocBegMagic, memory_order_release);
      return;
    }

    uptr old = kAllocBegMagic;
    if (!atomic_compare_exchange_strong(&magic, &old, 0,
                                        memory_order_release)) {
      CHECK_EQ(old, kAllocBegMagic);
    }
  }
};

void FlushUnneededMemProfShadowMemory(uptr p, uptr size) {
  // Since memprof's mapping is compacting, the shadow chunk may be
  // not page-aligned, so we only flush the page-aligned portion.
  ReleaseMemoryPagesToOS(MemToShadow(p), MemToShadow(p + size));
}

void MemprofMapUnmapCallback::OnMap(uptr p, uptr size) const {
  // Statistics.
  MemprofStats &thread_stats = GetCurrentThreadStats();
  thread_stats.mmaps++;
  thread_stats.mmaped += size;
}
void MemprofMapUnmapCallback::OnUnmap(uptr p, uptr size) const {
  // We are about to unmap a chunk of user memory.
  // Mark the corresponding shadow memory as not needed.
  FlushUnneededMemProfShadowMemory(p, size);
  // Statistics.
  MemprofStats &thread_stats = GetCurrentThreadStats();
  thread_stats.munmaps++;
  thread_stats.munmaped += size;
}

AllocatorCache *GetAllocatorCache(MemprofThreadLocalMallocStorage *ms) {
  CHECK(ms);
  return &ms->allocator_cache;
}

struct MemInfoBlock {
  u32 alloc_count;
  u64 total_access_count, min_access_count, max_access_count;
  u64 total_size;
  u32 min_size, max_size;
  u32 alloc_timestamp, dealloc_timestamp;
  u64 total_lifetime;
  u32 min_lifetime, max_lifetime;
  u32 alloc_cpu_id, dealloc_cpu_id;
  u32 num_migrated_cpu;

  // Only compared to prior deallocated object currently.
  u32 num_lifetime_overlaps;
  u32 num_same_alloc_cpu;
  u32 num_same_dealloc_cpu;

  u64 data_type_id; // TODO: hash of type name

  MemInfoBlock() : alloc_count(0) {}

  MemInfoBlock(u32 size, u64 access_count, u32 alloc_timestamp,
               u32 dealloc_timestamp, u32 alloc_cpu, u32 dealloc_cpu)
      : alloc_count(1), total_access_count(access_count),
        min_access_count(access_count), max_access_count(access_count),
        total_size(size), min_size(size), max_size(size),
        alloc_timestamp(alloc_timestamp), dealloc_timestamp(dealloc_timestamp),
        total_lifetime(dealloc_timestamp - alloc_timestamp),
        min_lifetime(total_lifetime), max_lifetime(total_lifetime),
        alloc_cpu_id(alloc_cpu), dealloc_cpu_id(dealloc_cpu),
        num_lifetime_overlaps(0), num_same_alloc_cpu(0),
        num_same_dealloc_cpu(0) {
    num_migrated_cpu = alloc_cpu_id != dealloc_cpu_id;
  }

  void Print(u64 id) {
    u64 p;
    if (flags()->print_terse) {
      p = total_size * 100 / alloc_count;
      Printf("MIB:%llu/%u/%d.%02d/%u/%u/", id, alloc_count, p / 100, p % 100,
             min_size, max_size);
      p = total_access_count * 100 / alloc_count;
      Printf("%d.%02d/%u/%u/", p / 100, p % 100, min_access_count,
             max_access_count);
      p = total_lifetime * 100 / alloc_count;
      Printf("%d.%02d/%u/%u/", p / 100, p % 100, min_lifetime, max_lifetime);
      Printf("%u/%u/%u/%u\n", num_migrated_cpu, num_lifetime_overlaps,
             num_same_alloc_cpu, num_same_dealloc_cpu);
    } else {
      p = total_size * 100 / alloc_count;
      Printf("Memory allocation stack id = %llu\n", id);
      Printf("\talloc_count %u, size (ave/min/max) %d.%02d / %u / %u\n",
             alloc_count, p / 100, p % 100, min_size, max_size);
      p = total_access_count * 100 / alloc_count;
      Printf("\taccess_count (ave/min/max): %d.%02d / %u / %u\n", p / 100,
             p % 100, min_access_count, max_access_count);
      p = total_lifetime * 100 / alloc_count;
      Printf("\tlifetime (ave/min/max): %d.%02d / %u / %u\n", p / 100, p % 100,
             min_lifetime, max_lifetime);
      Printf("\tnum migrated: %u, num lifetime overlaps: %u, num same alloc "
             "cpu: %u, num same dealloc_cpu: %u\n",
             num_migrated_cpu, num_lifetime_overlaps, num_same_alloc_cpu,
             num_same_dealloc_cpu);
    }
  }

  static void printHeader() {
    CHECK(flags()->print_terse);
    Printf("MIB:StackID/AllocCount/AveSize/MinSize/MaxSize/AveAccessCount/"
           "MinAccessCount/MaxAccessCount/AveLifetime/MinLifetime/MaxLifetime/"
           "NumMigratedCpu/NumLifetimeOverlaps/NumSameAllocCpu/"
           "NumSameDeallocCpu\n");
  }

  void Merge(MemInfoBlock &newMIB) {
    alloc_count += newMIB.alloc_count;

    total_access_count += newMIB.total_access_count;
    min_access_count = Min(min_access_count, newMIB.min_access_count);
    max_access_count = Max(max_access_count, newMIB.max_access_count);

    total_size += newMIB.total_size;
    min_size = Min(min_size, newMIB.min_size);
    max_size = Max(max_size, newMIB.max_size);

    total_lifetime += newMIB.total_lifetime;
    min_lifetime = Min(min_lifetime, newMIB.min_lifetime);
    max_lifetime = Max(max_lifetime, newMIB.max_lifetime);

    // We know newMIB was deallocated later, so just need to check if it was
    // allocated before last one deallocated.
    num_lifetime_overlaps += newMIB.alloc_timestamp < dealloc_timestamp;
    alloc_timestamp = newMIB.alloc_timestamp;
    dealloc_timestamp = newMIB.dealloc_timestamp;

    num_same_alloc_cpu += alloc_cpu_id == newMIB.alloc_cpu_id;
    num_same_dealloc_cpu += dealloc_cpu_id == newMIB.dealloc_cpu_id;
    alloc_cpu_id = newMIB.alloc_cpu_id;
    dealloc_cpu_id = newMIB.dealloc_cpu_id;
  }
};

static u32 AccessCount = 0;
static u32 MissCount = 0;

struct SetEntry {
  SetEntry() : id(0), MIB() {}
  bool Empty() { return id == 0; }
  void Print() {
    CHECK(!Empty());
    MIB.Print(id);
  }
  // The stack id
  u64 id;
  MemInfoBlock MIB;
};

struct CacheSet {
  enum { kSetSize = 4 };

  void PrintAll() {
    for (int i = 0; i < kSetSize; i++) {
      if (Entries[i].Empty())
        continue;
      Entries[i].Print();
    }
  }
  void insertOrMerge(u64 new_id, MemInfoBlock &newMIB) {
    AccessCount++;
    SetAccessCount++;

    for (int i = 0; i < kSetSize; i++) {
      auto id = Entries[i].id;
      // Check if this is a hit or an empty entry. Since we always move any
      // filled locations to the front of the array (see below), we don't need
      // to look after finding the first empty entry.
      if (id == new_id || !id) {
        if (id == 0) {
          Entries[i].id = new_id;
          Entries[i].MIB = newMIB;
        } else {
          Entries[i].MIB.Merge(newMIB);
        }
        // Assuming some id locality, we try to swap the matching entry
        // into the first set position.
        if (i != 0) {
          auto tmp = Entries[0];
          Entries[0] = Entries[i];
          Entries[i] = tmp;
        }
        return;
      }
    }

    // Miss
    MissCount++;
    SetMissCount++;

    // We try to find the entries with the lowest alloc count to be evicted:
    int min_idx = 0;
    u64 min_count = Entries[0].MIB.alloc_count;
    for (int i = 1; i < kSetSize; i++) {
      CHECK(!Entries[i].Empty());
      if (Entries[i].MIB.alloc_count < min_count) {
        min_idx = i;
        min_count = Entries[i].MIB.alloc_count;
      }
    }

    // Print the evicted entry profile information
    if (!flags()->print_terse)
      Printf("Evicted:\n");
    Entries[min_idx].Print();

    // Similar to the hit case, put new MIB in first set position.
    if (min_idx != 0)
      Entries[min_idx] = Entries[0];
    Entries[0].id = new_id;
    Entries[0].MIB = newMIB;
  }

  void PrintMissRate(int i) {
    u64 p = SetAccessCount ? SetMissCount * 10000ULL / SetAccessCount : 0;
    Printf("Set %d miss rate: %d / %d = %5d.%02d%%\n", i, SetMissCount,
           SetAccessCount, p / 100, p % 100);
  }

  SetEntry Entries[kSetSize];
  u32 SetAccessCount = 0;
  u32 SetMissCount = 0;
};

struct MemInfoBlockCache {
  MemInfoBlockCache() {
    if (common_flags()->print_module_map)
      DumpProcessMap();
    if (flags()->print_terse)
      MemInfoBlock::printHeader();
    Sets =
        (CacheSet *)malloc(sizeof(CacheSet) * flags()->mem_info_cache_entries);
    Constructed = true;
  }

  ~MemInfoBlockCache() { free(Sets); }

  void insertOrMerge(u64 new_id, MemInfoBlock &newMIB) {
    u64 hv = new_id;

    // Use mod method where number of entries should be a prime close to power
    // of 2.
    hv %= flags()->mem_info_cache_entries;

    return Sets[hv].insertOrMerge(new_id, newMIB);
  }

  void PrintAll() {
    for (int i = 0; i < flags()->mem_info_cache_entries; i++) {
      Sets[i].PrintAll();
    }
  }

  void PrintMissRate() {
    if (!flags()->print_mem_info_cache_miss_rate)
      return;
    u64 p = AccessCount ? MissCount * 10000ULL / AccessCount : 0;
    Printf("Overall miss rate: %d / %d = %5d.%02d%%\n", MissCount, AccessCount,
           p / 100, p % 100);
    if (flags()->print_mem_info_cache_miss_rate_details)
      for (int i = 0; i < flags()->mem_info_cache_entries; i++)
        Sets[i].PrintMissRate(i);
  }

  CacheSet *Sets;
  // Flag when the Sets have been allocated, in case a deallocation is called
  // very early before the static init of the Allocator and therefore this table
  // have completed.
  bool Constructed = false;
};

// Accumulates the access count from the shadow for the given pointer and size.
u64 GetShadowCount(uptr p, u32 size) {
  u64 *shadow = (u64 *)MEM_TO_SHADOW(p);
  u64 *shadow_end = (u64 *)MEM_TO_SHADOW(p + size);
  u64 count = 0;
  for (; shadow <= shadow_end; shadow++)
    count += *shadow;
  return count;
}

// Clears the shadow counters (when memory is allocated).
void ClearShadow(uptr addr, uptr size) {
  CHECK(AddrIsAlignedByGranularity(addr));
  CHECK(AddrIsInMem(addr));
  CHECK(AddrIsAlignedByGranularity(addr + size));
  CHECK(AddrIsInMem(addr + size - SHADOW_GRANULARITY));
  CHECK(REAL(memset));
  uptr shadow_beg = MEM_TO_SHADOW(addr);
  uptr shadow_end = MEM_TO_SHADOW(addr + size - SHADOW_GRANULARITY) + 1;
  if (shadow_end - shadow_beg < common_flags()->clear_shadow_mmap_threshold) {
    REAL(memset)((void *)shadow_beg, 0, shadow_end - shadow_beg);
  } else {
    uptr page_size = GetPageSizeCached();
    uptr page_beg = RoundUpTo(shadow_beg, page_size);
    uptr page_end = RoundDownTo(shadow_end, page_size);

    if (page_beg >= page_end) {
      REAL(memset)((void *)shadow_beg, 0, shadow_end - shadow_beg);
    } else {
      if (page_beg != shadow_beg) {
        REAL(memset)((void *)shadow_beg, 0, page_beg - shadow_beg);
      }
      if (page_end != shadow_end) {
        REAL(memset)((void *)page_end, 0, shadow_end - page_end);
      }
      ReserveShadowMemoryRange(page_beg, page_end - 1, nullptr);
    }
  }
}

struct Allocator {
  static const uptr kMaxAllowedMallocSize = 1ULL << kMaxAllowedMallocBits;

  MemprofAllocator allocator;
  StaticSpinMutex fallback_mutex;
  AllocatorCache fallback_allocator_cache;

  uptr max_user_defined_malloc_size;
  atomic_uint8_t rss_limit_exceeded;

  MemInfoBlockCache MemInfoBlockTable;
  bool destructing;

  // ------------------- Initialization ------------------------
  explicit Allocator(LinkerInitialized) : destructing(false) {}

  ~Allocator() { FinishAndPrint(); }

  void FinishAndPrint() {
    if (!flags()->print_terse)
      Printf("Live on exit:\n");
    allocator.ForceLock();
    allocator.ForEachChunk(
        [](uptr chunk, void *alloc) {
          u64 user_requested_size;
          MemprofChunk *m =
              ((Allocator *)alloc)
                  ->GetMemprofChunk((void *)chunk, user_requested_size);
          if (!m)
            return;
          uptr user_beg = ((uptr)m) + kChunkHeaderSize;
          u64 c = GetShadowCount(user_beg, user_requested_size);
          long curtime = GetTimestamp();
          MemInfoBlock newMIB(user_requested_size, c, m->timestamp_ms, curtime,
                              m->cpu_id, GetCpuId());
          ((Allocator *)alloc)
              ->MemInfoBlockTable.insertOrMerge(m->alloc_context_id, newMIB);
        },
        this);
    allocator.ForceUnlock();

    destructing = true;
    MemInfoBlockTable.PrintMissRate();
    MemInfoBlockTable.PrintAll();
    StackDepotPrintAll();
  }

  void InitLinkerInitialized() {
    SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
    allocator.InitLinkerInitialized(
        common_flags()->allocator_release_to_os_interval_ms);
    max_user_defined_malloc_size = common_flags()->max_allocation_size_mb
                                       ? common_flags()->max_allocation_size_mb
                                             << 20
                                       : kMaxAllowedMallocSize;
  }

  bool RssLimitExceeded() {
    return atomic_load(&rss_limit_exceeded, memory_order_relaxed);
  }

  void SetRssLimitExceeded(bool limit_exceeded) {
    atomic_store(&rss_limit_exceeded, limit_exceeded, memory_order_relaxed);
  }

  // -------------------- Allocation/Deallocation routines ---------------
  void *Allocate(uptr size, uptr alignment, BufferedStackTrace *stack,
                 AllocType alloc_type) {
    if (UNLIKELY(!memprof_inited))
      MemprofInitFromRtl();
    if (RssLimitExceeded()) {
      if (AllocatorMayReturnNull())
        return nullptr;
      ReportRssLimitExceeded(stack);
    }
    CHECK(stack);
    const uptr min_alignment = MEMPROF_ALIGNMENT;
    if (alignment < min_alignment)
      alignment = min_alignment;
    if (size == 0) {
      // We'd be happy to avoid allocating memory for zero-size requests, but
      // some programs/tests depend on this behavior and assume that malloc
      // would not return NULL even for zero-size allocations. Moreover, it
      // looks like operator new should never return NULL, and results of
      // consecutive "new" calls must be different even if the allocated size
      // is zero.
      size = 1;
    }
    CHECK(IsPowerOfTwo(alignment));
    uptr rounded_size = RoundUpTo(size, alignment);
    uptr needed_size = rounded_size + kChunkHeaderSize;
    if (alignment > min_alignment)
      needed_size += alignment;
    CHECK(IsAligned(needed_size, min_alignment));
    if (size > kMaxAllowedMallocSize || needed_size > kMaxAllowedMallocSize ||
        size > max_user_defined_malloc_size) {
      if (AllocatorMayReturnNull()) {
        Report("WARNING: MemProfiler failed to allocate 0x%zx bytes\n",
               (void *)size);
        return nullptr;
      }
      uptr malloc_limit =
          Min(kMaxAllowedMallocSize, max_user_defined_malloc_size);
      ReportAllocationSizeTooBig(size, malloc_limit, stack);
    }

    MemprofThread *t = GetCurrentThread();
    void *allocated;
    if (t) {
      AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
      allocated = allocator.Allocate(cache, needed_size, 8);
    } else {
      SpinMutexLock l(&fallback_mutex);
      AllocatorCache *cache = &fallback_allocator_cache;
      allocated = allocator.Allocate(cache, needed_size, 8);
    }
    if (UNLIKELY(!allocated)) {
      SetAllocatorOutOfMemory();
      if (AllocatorMayReturnNull())
        return nullptr;
      ReportOutOfMemory(size, stack);
    }

    uptr alloc_beg = reinterpret_cast<uptr>(allocated);
    uptr alloc_end = alloc_beg + needed_size;
    uptr beg_plus_header = alloc_beg + kChunkHeaderSize;
    uptr user_beg = beg_plus_header;
    if (!IsAligned(user_beg, alignment))
      user_beg = RoundUpTo(user_beg, alignment);
    uptr user_end = user_beg + size;
    CHECK_LE(user_end, alloc_end);
    uptr chunk_beg = user_beg - kChunkHeaderSize;
    MemprofChunk *m = reinterpret_cast<MemprofChunk *>(chunk_beg);
    m->from_memalign = alloc_beg != chunk_beg;
    CHECK(size);

    m->cpu_id = GetCpuId();
    m->timestamp_ms = GetTimestamp();
    m->alloc_context_id = StackDepotPut(*stack);

    uptr size_rounded_down_to_granularity =
        RoundDownTo(size, SHADOW_GRANULARITY);
    if (size_rounded_down_to_granularity)
      ClearShadow(user_beg, size_rounded_down_to_granularity);

    MemprofStats &thread_stats = GetCurrentThreadStats();
    thread_stats.mallocs++;
    thread_stats.malloced += size;
    thread_stats.malloced_overhead += needed_size - size;
    if (needed_size > SizeClassMap::kMaxSize)
      thread_stats.malloc_large++;
    else
      thread_stats.malloced_by_size[SizeClassMap::ClassID(needed_size)]++;

    void *res = reinterpret_cast<void *>(user_beg);
    atomic_store(&m->user_requested_size, size, memory_order_release);
    if (alloc_beg != chunk_beg) {
      CHECK_LE(alloc_beg + sizeof(LargeChunkHeader), chunk_beg);
      reinterpret_cast<LargeChunkHeader *>(alloc_beg)->Set(m);
    }
    MEMPROF_MALLOC_HOOK(res, size);
    return res;
  }

  void Deallocate(void *ptr, uptr delete_size, uptr delete_alignment,
                  BufferedStackTrace *stack, AllocType alloc_type) {
    uptr p = reinterpret_cast<uptr>(ptr);
    if (p == 0)
      return;

    MEMPROF_FREE_HOOK(ptr);

    uptr chunk_beg = p - kChunkHeaderSize;
    MemprofChunk *m = reinterpret_cast<MemprofChunk *>(chunk_beg);

    u64 user_requested_size =
        atomic_exchange(&m->user_requested_size, 0, memory_order_acquire);
    if (memprof_inited && memprof_init_done && !destructing &&
        MemInfoBlockTable.Constructed) {
      u64 c = GetShadowCount(p, user_requested_size);
      long curtime = GetTimestamp();

      MemInfoBlock newMIB(user_requested_size, c, m->timestamp_ms, curtime,
                          m->cpu_id, GetCpuId());
      {
        SpinMutexLock l(&fallback_mutex);
        MemInfoBlockTable.insertOrMerge(m->alloc_context_id, newMIB);
      }
    }

    MemprofStats &thread_stats = GetCurrentThreadStats();
    thread_stats.frees++;
    thread_stats.freed += user_requested_size;

    void *alloc_beg = m->AllocBeg();
    if (alloc_beg != m) {
      // Clear the magic value, as allocator internals may overwrite the
      // contents of deallocated chunk, confusing GetMemprofChunk lookup.
      reinterpret_cast<LargeChunkHeader *>(alloc_beg)->Set(nullptr);
    }

    MemprofThread *t = GetCurrentThread();
    if (t) {
      AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
      allocator.Deallocate(cache, alloc_beg);
    } else {
      SpinMutexLock l(&fallback_mutex);
      AllocatorCache *cache = &fallback_allocator_cache;
      allocator.Deallocate(cache, alloc_beg);
    }
  }

  void *Reallocate(void *old_ptr, uptr new_size, BufferedStackTrace *stack) {
    CHECK(old_ptr && new_size);
    uptr p = reinterpret_cast<uptr>(old_ptr);
    uptr chunk_beg = p - kChunkHeaderSize;
    MemprofChunk *m = reinterpret_cast<MemprofChunk *>(chunk_beg);

    MemprofStats &thread_stats = GetCurrentThreadStats();
    thread_stats.reallocs++;
    thread_stats.realloced += new_size;

    void *new_ptr = Allocate(new_size, 8, stack, FROM_MALLOC);
    if (new_ptr) {
      CHECK_NE(REAL(memcpy), nullptr);
      uptr memcpy_size = Min(new_size, m->UsedSize());
      REAL(memcpy)(new_ptr, old_ptr, memcpy_size);
      Deallocate(old_ptr, 0, 0, stack, FROM_MALLOC);
    }
    return new_ptr;
  }

  void *Calloc(uptr nmemb, uptr size, BufferedStackTrace *stack) {
    if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
      if (AllocatorMayReturnNull())
        return nullptr;
      ReportCallocOverflow(nmemb, size, stack);
    }
    void *ptr = Allocate(nmemb * size, 8, stack, FROM_MALLOC);
    // If the memory comes from the secondary allocator no need to clear it
    // as it comes directly from mmap.
    if (ptr && allocator.FromPrimary(ptr))
      REAL(memset)(ptr, 0, nmemb * size);
    return ptr;
  }

  void CommitBack(MemprofThreadLocalMallocStorage *ms,
                  BufferedStackTrace *stack) {
    AllocatorCache *ac = GetAllocatorCache(ms);
    allocator.SwallowCache(ac);
  }

  // -------------------------- Chunk lookup ----------------------

  // Assumes alloc_beg == allocator.GetBlockBegin(alloc_beg).
  MemprofChunk *GetMemprofChunk(void *alloc_beg, u64 &user_requested_size) {
    if (!alloc_beg)
      return nullptr;
    MemprofChunk *p = reinterpret_cast<LargeChunkHeader *>(alloc_beg)->Get();
    if (!p) {
      if (!allocator.FromPrimary(alloc_beg))
        return nullptr;
      p = reinterpret_cast<MemprofChunk *>(alloc_beg);
    }
    // The size is reset to 0 on deallocation (and a min of 1 on
    // allocation).
    user_requested_size =
        atomic_load(&p->user_requested_size, memory_order_acquire);
    if (user_requested_size)
      return p;
    return nullptr;
  }

  MemprofChunk *GetMemprofChunkByAddr(uptr p, u64 &user_requested_size) {
    void *alloc_beg = allocator.GetBlockBegin(reinterpret_cast<void *>(p));
    return GetMemprofChunk(alloc_beg, user_requested_size);
  }

  uptr AllocationSize(uptr p) {
    u64 user_requested_size;
    MemprofChunk *m = GetMemprofChunkByAddr(p, user_requested_size);
    if (!m)
      return 0;
    if (m->Beg() != p)
      return 0;
    return user_requested_size;
  }

  void Purge(BufferedStackTrace *stack) { allocator.ForceReleaseToOS(); }

  void PrintStats() { allocator.PrintStats(); }

  void ForceLock() NO_THREAD_SAFETY_ANALYSIS {
    allocator.ForceLock();
    fallback_mutex.Lock();
  }

  void ForceUnlock() NO_THREAD_SAFETY_ANALYSIS {
    fallback_mutex.Unlock();
    allocator.ForceUnlock();
  }
};

static Allocator instance(LINKER_INITIALIZED);

static MemprofAllocator &get_allocator() { return instance.allocator; }

void InitializeAllocator() { instance.InitLinkerInitialized(); }

void MemprofThreadLocalMallocStorage::CommitBack() {
  GET_STACK_TRACE_MALLOC;
  instance.CommitBack(this, &stack);
}

void PrintInternalAllocatorStats() { instance.PrintStats(); }

void memprof_free(void *ptr, BufferedStackTrace *stack, AllocType alloc_type) {
  instance.Deallocate(ptr, 0, 0, stack, alloc_type);
}

void memprof_delete(void *ptr, uptr size, uptr alignment,
                    BufferedStackTrace *stack, AllocType alloc_type) {
  instance.Deallocate(ptr, size, alignment, stack, alloc_type);
}

void *memprof_malloc(uptr size, BufferedStackTrace *stack) {
  return SetErrnoOnNull(instance.Allocate(size, 8, stack, FROM_MALLOC));
}

void *memprof_calloc(uptr nmemb, uptr size, BufferedStackTrace *stack) {
  return SetErrnoOnNull(instance.Calloc(nmemb, size, stack));
}

void *memprof_reallocarray(void *p, uptr nmemb, uptr size,
                           BufferedStackTrace *stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportReallocArrayOverflow(nmemb, size, stack);
  }
  return memprof_realloc(p, nmemb * size, stack);
}

void *memprof_realloc(void *p, uptr size, BufferedStackTrace *stack) {
  if (!p)
    return SetErrnoOnNull(instance.Allocate(size, 8, stack, FROM_MALLOC));
  if (size == 0) {
    if (flags()->allocator_frees_and_returns_null_on_realloc_zero) {
      instance.Deallocate(p, 0, 0, stack, FROM_MALLOC);
      return nullptr;
    }
    // Allocate a size of 1 if we shouldn't free() on Realloc to 0
    size = 1;
  }
  return SetErrnoOnNull(instance.Reallocate(p, size, stack));
}

void *memprof_valloc(uptr size, BufferedStackTrace *stack) {
  return SetErrnoOnNull(
      instance.Allocate(size, GetPageSizeCached(), stack, FROM_MALLOC));
}

void *memprof_pvalloc(uptr size, BufferedStackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportPvallocOverflow(size, stack);
  }
  // pvalloc(0) should allocate one page.
  size = size ? RoundUpTo(size, PageSize) : PageSize;
  return SetErrnoOnNull(instance.Allocate(size, PageSize, stack, FROM_MALLOC));
}

void *memprof_memalign(uptr alignment, uptr size, BufferedStackTrace *stack,
                       AllocType alloc_type) {
  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAllocationAlignment(alignment, stack);
  }
  return SetErrnoOnNull(instance.Allocate(size, alignment, stack, alloc_type));
}

void *memprof_aligned_alloc(uptr alignment, uptr size,
                            BufferedStackTrace *stack) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(alignment, size))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAlignedAllocAlignment(size, alignment, stack);
  }
  return SetErrnoOnNull(instance.Allocate(size, alignment, stack, FROM_MALLOC));
}

int memprof_posix_memalign(void **memptr, uptr alignment, uptr size,
                           BufferedStackTrace *stack) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    ReportInvalidPosixMemalignAlignment(alignment, stack);
  }
  void *ptr = instance.Allocate(size, alignment, stack, FROM_MALLOC);
  if (UNLIKELY(!ptr))
    // OOM error is already taken care of by Allocate.
    return errno_ENOMEM;
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

uptr memprof_malloc_usable_size(const void *ptr, uptr pc, uptr bp) {
  if (!ptr)
    return 0;
  uptr usable_size = instance.AllocationSize(reinterpret_cast<uptr>(ptr));
  return usable_size;
}

void MemprofSoftRssLimitExceededCallback(bool limit_exceeded) {
  instance.SetRssLimitExceeded(limit_exceeded);
}

} // namespace __memprof

// ---------------------- Interface ---------------- {{{1
using namespace __memprof;

#if !SANITIZER_SUPPORTS_WEAK_HOOKS
// Provide default (no-op) implementation of malloc hooks.
SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_malloc_hook, void *ptr,
                             uptr size) {
  (void)ptr;
  (void)size;
}

SANITIZER_INTERFACE_WEAK_DEF(void, __sanitizer_free_hook, void *ptr) {
  (void)ptr;
}
#endif

uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

int __sanitizer_get_ownership(const void *p) {
  return memprof_malloc_usable_size(p, 0, 0) != 0;
}

uptr __sanitizer_get_allocated_size(const void *p) {
  return memprof_malloc_usable_size(p, 0, 0);
}

int __memprof_profile_dump() {
  instance.FinishAndPrint();
  // In the future we may want to return non-zero if there are any errors
  // detected during the dumping process.
  return 0;
}
