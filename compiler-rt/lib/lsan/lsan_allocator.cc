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
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include "lsan_common.h"

extern "C" void *memset(void *ptr, int value, uptr num);

namespace __lsan {

static const uptr kMaxAllowedMallocSize = 8UL << 30;
static const uptr kAllocatorSpace = 0x600000000000ULL;
static const uptr kAllocatorSize  =  0x40000000000ULL;  // 4T.

struct ChunkMetadata {
  bool allocated : 8;  // Must be first.
  ChunkTag tag : 2;
  uptr requested_size : 54;
  u32 stack_trace_id;
};

typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize,
        sizeof(ChunkMetadata), DefaultSizeClassMap> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
          SecondaryAllocator> Allocator;

static Allocator allocator;
static THREADLOCAL AllocatorCache cache;

void InitializeAllocator() {
  allocator.Init();
}

void AllocatorThreadFinish() {
  allocator.SwallowCache(&cache);
}

static ChunkMetadata *Metadata(void *p) {
  return reinterpret_cast<ChunkMetadata *>(allocator.GetMetaData(p));
}

static void RegisterAllocation(const StackTrace &stack, void *p, uptr size) {
  if (!p) return;
  ChunkMetadata *m = Metadata(p);
  CHECK(m);
  m->tag = DisabledInThisThread() ? kIgnored : kDirectlyLeaked;
  m->stack_trace_id = StackDepotPut(stack.trace, stack.size);
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
    return 0;
  }
  void *p = allocator.Allocate(&cache, size, alignment, false);
  // Do not rely on the allocator to clear the memory (it's slow).
  if (cleared && allocator.FromPrimary(p))
    memset(p, 0, size);
  RegisterAllocation(stack, p, size);
  return p;
}

void Deallocate(void *p) {
  RegisterDeallocation(p);
  allocator.Deallocate(&cache, p);
}

void *Reallocate(const StackTrace &stack, void *p, uptr new_size,
                 uptr alignment) {
  RegisterDeallocation(p);
  if (new_size > kMaxAllowedMallocSize) {
    Report("WARNING: LeakSanitizer failed to allocate %zu bytes\n", new_size);
    allocator.Deallocate(&cache, p);
    return 0;
  }
  p = allocator.Reallocate(&cache, p, new_size, alignment);
  RegisterAllocation(stack, p, new_size);
  return p;
}

void GetAllocatorCacheRange(uptr *begin, uptr *end) {
  *begin = (uptr)&cache;
  *end = *begin + sizeof(cache);
}

uptr GetMallocUsableSize(void *p) {
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
  if (m->allocated && addr < chunk + m->requested_size)
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
}  // namespace __lsan
