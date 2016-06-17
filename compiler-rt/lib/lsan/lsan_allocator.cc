//=-- lsan_allocator.cc ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// See lsan_allocator.h for details.
//
//===----------------------------------------------------------------------===//

#include "lsan_allocator.h"

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "lsan_common.h"

extern "C" void *memset(void *ptr, int value, uptr num);

namespace __lsan {

struct ChunkMetadata {
  u8 allocated : 8;  // Must be first.
  ChunkTag tag : 2;
  uptr requested_size : 54;
  u32 stack_trace_id;
};

#if defined(__mips64) || defined(__aarch64__)
static const uptr kMaxAllowedMallocSize = 4UL << 30;
static const uptr kRegionSizeLog = 20;
static const uptr kNumRegions = SANITIZER_MMAP_RANGE_SIZE >> kRegionSizeLog;
typedef TwoLevelByteMap<(kNumRegions >> 12), 1 << 12> ByteMap;
typedef CompactSizeClassMap SizeClassMap;
typedef SizeClassAllocator32<0, SANITIZER_MMAP_RANGE_SIZE,
    sizeof(ChunkMetadata), SizeClassMap, kRegionSizeLog, ByteMap>
    PrimaryAllocator;
#else
static const uptr kMaxAllowedMallocSize = 8UL << 30;
static const uptr kAllocatorSpace = 0x600000000000ULL;
static const uptr kAllocatorSize = 0x40000000000ULL; // 4T.
typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize,
        sizeof(ChunkMetadata), DefaultSizeClassMap> PrimaryAllocator;
#endif
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
          SecondaryAllocator> Allocator;

static Allocator allocator;
static THREADLOCAL AllocatorCache cache;

void InitializeAllocator() {
  allocator.InitLinkerInitialized(common_flags()->allocator_may_return_null);
}

void AllocatorThreadFinish() {
  allocator.SwallowCache(&cache);
}

static ChunkMetadata *Metadata(const void *p) {
  return reinterpret_cast<ChunkMetadata *>(allocator.GetMetaData(p));
}

static void RegisterAllocation(const StackTrace &stack, void *p, uptr size) {
  if (!p) return;
  ChunkMetadata *m = Metadata(p);
  CHECK(m);
  m->tag = DisabledInThisThread() ? kIgnored : kDirectlyLeaked;
  m->stack_trace_id = StackDepotPut(stack);
  m->requested_size = size;
  atomic_store(reinterpret_cast<atomic_uint8_t *>(m), 1, memory_order_relaxed);
}

static void RegisterDeallocation(void *p) {
  if (!p) return;
  ChunkMetadata *m = Metadata(p);
  CHECK(m);
  atomic_store(reinterpret_cast<atomic_uint8_t *>(m), 0, memory_order_relaxed);
}

void *Allocate(const StackTrace &stack, uptr size, uptr alignment,
               bool cleared) {
  if (size == 0)
    size = 1;
  if (size > kMaxAllowedMallocSize) {
    Report("WARNING: LeakSanitizer failed to allocate %zu bytes\n", size);
    return nullptr;
  }
  void *p = allocator.Allocate(&cache, size, alignment, false);
  // Do not rely on the allocator to clear the memory (it's slow).
  if (cleared && allocator.FromPrimary(p))
    memset(p, 0, size);
  RegisterAllocation(stack, p, size);
  if (&__sanitizer_malloc_hook) __sanitizer_malloc_hook(p, size);
  RunMallocHooks(p, size);
  return p;
}

void Deallocate(void *p) {
  if (&__sanitizer_free_hook) __sanitizer_free_hook(p);
  RunFreeHooks(p);
  RegisterDeallocation(p);
  allocator.Deallocate(&cache, p);
}

void *Reallocate(const StackTrace &stack, void *p, uptr new_size,
                 uptr alignment) {
  RegisterDeallocation(p);
  if (new_size > kMaxAllowedMallocSize) {
    Report("WARNING: LeakSanitizer failed to allocate %zu bytes\n", new_size);
    allocator.Deallocate(&cache, p);
    return nullptr;
  }
  p = allocator.Reallocate(&cache, p, new_size, alignment);
  RegisterAllocation(stack, p, new_size);
  return p;
}

void GetAllocatorCacheRange(uptr *begin, uptr *end) {
  *begin = (uptr)&cache;
  *end = *begin + sizeof(cache);
}

uptr GetMallocUsableSize(const void *p) {
  ChunkMetadata *m = Metadata(p);
  if (!m) return 0;
  return m->requested_size;
}

///// Interface to the common LSan module. /////

void LockAllocator() {
  allocator.ForceLock();
}

void UnlockAllocator() {
  allocator.ForceUnlock();
}

void GetAllocatorGlobalRange(uptr *begin, uptr *end) {
  *begin = (uptr)&allocator;
  *end = *begin + sizeof(allocator);
}

uptr PointsIntoChunk(void* p) {
  uptr addr = reinterpret_cast<uptr>(p);
  uptr chunk = reinterpret_cast<uptr>(allocator.GetBlockBeginFastLocked(p));
  if (!chunk) return 0;
  // LargeMmapAllocator considers pointers to the meta-region of a chunk to be
  // valid, but we don't want that.
  if (addr < chunk) return 0;
  ChunkMetadata *m = Metadata(reinterpret_cast<void *>(chunk));
  CHECK(m);
  if (!m->allocated)
    return 0;
  if (addr < chunk + m->requested_size)
    return chunk;
  if (IsSpecialCaseOfOperatorNew0(chunk, m->requested_size, addr))
    return chunk;
  return 0;
}

uptr GetUserBegin(uptr chunk) {
  return chunk;
}

LsanMetadata::LsanMetadata(uptr chunk) {
  metadata_ = Metadata(reinterpret_cast<void *>(chunk));
  CHECK(metadata_);
}

bool LsanMetadata::allocated() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->allocated;
}

ChunkTag LsanMetadata::tag() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->tag;
}

void LsanMetadata::set_tag(ChunkTag value) {
  reinterpret_cast<ChunkMetadata *>(metadata_)->tag = value;
}

uptr LsanMetadata::requested_size() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->requested_size;
}

u32 LsanMetadata::stack_trace_id() const {
  return reinterpret_cast<ChunkMetadata *>(metadata_)->stack_trace_id;
}

void ForEachChunk(ForEachChunkCallback callback, void *arg) {
  allocator.ForEachChunk(callback, arg);
}

IgnoreObjectResult IgnoreObjectLocked(const void *p) {
  void *chunk = allocator.GetBlockBegin(p);
  if (!chunk || p < chunk) return kIgnoreObjectInvalid;
  ChunkMetadata *m = Metadata(chunk);
  CHECK(m);
  if (m->allocated && (uptr)p < (uptr)chunk + m->requested_size) {
    if (m->tag == kIgnored)
      return kIgnoreObjectAlreadyIgnored;
    m->tag = kIgnored;
    return kIgnoreObjectSuccess;
  } else {
    return kIgnoreObjectInvalid;
  }
}
} // namespace __lsan

using namespace __lsan;

extern "C" {
SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatAllocated];
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatMapped];
}

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_free_bytes() { return 0; }

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_unmapped_bytes() { return 0; }

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

SANITIZER_INTERFACE_ATTRIBUTE
int __sanitizer_get_ownership(const void *p) { return Metadata(p) != nullptr; }

SANITIZER_INTERFACE_ATTRIBUTE
uptr __sanitizer_get_allocated_size(const void *p) {
  return GetMallocUsableSize(p);
}
} // extern "C"
