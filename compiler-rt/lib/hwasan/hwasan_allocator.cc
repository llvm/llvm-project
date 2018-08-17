//===-- hwasan_allocator.cc ------------------------- ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
// HWAddressSanitizer allocator.
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_checks.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"
#include "sanitizer_common/sanitizer_allocator_report.h"
#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "hwasan.h"
#include "hwasan_allocator.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_poisoning.h"

namespace __hwasan {

enum {
  CHUNK_INVALID = 0,
  CHUNK_FREE = 1,
  CHUNK_ALLOCATED = 2
};

struct Metadata {
  u64 state : 2;
  u64 requested_size : 62;
  u32 alloc_context_id;
  u32 free_context_id;
};

bool HwasanChunkView::IsValid() const {
  return metadata_ && metadata_->state != CHUNK_INVALID;
}
bool HwasanChunkView::IsAllocated() const {
  return metadata_ && metadata_->state == CHUNK_ALLOCATED;
}
uptr HwasanChunkView::Beg() const {
  return block_;
}
uptr HwasanChunkView::End() const {
  return Beg() + UsedSize();
}
uptr HwasanChunkView::UsedSize() const {
  return metadata_->requested_size;
}
u32 HwasanChunkView::GetAllocStackId() const {
  return metadata_->alloc_context_id;
}
u32 HwasanChunkView::GetFreeStackId() const {
  return metadata_->free_context_id;
}

struct HwasanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnUnmap(uptr p, uptr size) const {
    // We are about to unmap a chunk of user memory.
    // It can return as user-requested mmap() or another thread stack.
    // Make it accessible with zero-tagged pointer.
    TagMemory(p, size, 0);
  }
};

#if !defined(__aarch64__) && !defined(__x86_64__)
#error Unsupported platform
#endif

static const uptr kMaxAllowedMallocSize = 2UL << 30;  // 2G
static const uptr kRegionSizeLog = 20;
static const uptr kNumRegions = SANITIZER_MMAP_RANGE_SIZE >> kRegionSizeLog;
typedef TwoLevelByteMap<(kNumRegions >> 12), 1 << 12> ByteMap;

struct AP32 {
  static const uptr kSpaceBeg = 0;
  static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
  static const uptr kMetadataSize = sizeof(Metadata);
  typedef __sanitizer::CompactSizeClassMap SizeClassMap;
  static const uptr kRegionSizeLog = __hwasan::kRegionSizeLog;
  typedef __hwasan::ByteMap ByteMap;
  typedef HwasanMapUnmapCallback MapUnmapCallback;
  static const uptr kFlags = 0;
};
typedef SizeClassAllocator32<AP32> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<HwasanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
                          SecondaryAllocator> Allocator;

static Allocator allocator;
static AllocatorCache fallback_allocator_cache;
static SpinMutex fallback_mutex;
static atomic_uint8_t hwasan_allocator_tagging_enabled;

static const tag_t kFallbackAllocTag = 0xBB;
static const tag_t kFallbackFreeTag = 0xBC;

void HwasanAllocatorInit() {
  atomic_store_relaxed(&hwasan_allocator_tagging_enabled,
                       !flags()->disable_allocator_tagging);
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator.Init(common_flags()->allocator_release_to_os_interval_ms);
}

AllocatorCache *GetAllocatorCache(HwasanThreadLocalMallocStorage *ms) {
  CHECK(ms);
  CHECK_LE(sizeof(AllocatorCache), sizeof(ms->allocator_cache));
  return reinterpret_cast<AllocatorCache *>(ms->allocator_cache);
}

void HwasanThreadLocalMallocStorage::CommitBack() {
  allocator.SwallowCache(GetAllocatorCache(this));
}

static void *HwasanAllocate(StackTrace *stack, uptr size, uptr alignment,
                          bool zeroise) {
  alignment = Max(alignment, kShadowAlignment);
  size = RoundUpTo(size, kShadowAlignment);

  if (size > kMaxAllowedMallocSize) {
    if (AllocatorMayReturnNull()) {
      Report("WARNING: HWAddressSanitizer failed to allocate 0x%zx bytes\n",
             size);
      return nullptr;
    }
    ReportAllocationSizeTooBig(size, kMaxAllowedMallocSize, stack);
  }
  HwasanThread *t = GetCurrentThread();
  void *allocated;
  if (t) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocated = allocator.Allocate(cache, size, alignment);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocated = allocator.Allocate(cache, size, alignment);
  }
  if (UNLIKELY(!allocated)) {
    SetAllocatorOutOfMemory();
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportOutOfMemory(size, stack);
  }
  Metadata *meta =
      reinterpret_cast<Metadata *>(allocator.GetMetaData(allocated));
  meta->state = CHUNK_ALLOCATED;
  meta->requested_size = size;
  meta->alloc_context_id = StackDepotPut(*stack);
  if (zeroise) {
    internal_memset(allocated, 0, size);
  } else if (flags()->max_malloc_fill_size > 0) {
    uptr fill_size = Min(size, (uptr)flags()->max_malloc_fill_size);
    internal_memset(allocated, flags()->malloc_fill_byte, fill_size);
  }

  void *user_ptr = allocated;
  if (flags()->tag_in_malloc &&
      atomic_load_relaxed(&hwasan_allocator_tagging_enabled))
    user_ptr = (void *)TagMemoryAligned(
        (uptr)user_ptr, size, t ? t->GenerateRandomTag() : kFallbackAllocTag);

  HWASAN_MALLOC_HOOK(user_ptr, size);
  return user_ptr;
}

void HwasanDeallocate(StackTrace *stack, void *user_ptr) {
  CHECK(user_ptr);
  HWASAN_FREE_HOOK(user_ptr);

  void *p = GetAddressFromPointer(user_ptr);
  Metadata *meta = reinterpret_cast<Metadata *>(allocator.GetMetaData(p));
  uptr size = meta->requested_size;
  meta->state = CHUNK_FREE;
  meta->requested_size = 0;
  meta->free_context_id = StackDepotPut(*stack);
  // This memory will not be reused by anyone else, so we are free to keep it
  // poisoned.
  HwasanThread *t = GetCurrentThread();
  if (flags()->max_free_fill_size > 0) {
    uptr fill_size = Min(size, (uptr)flags()->max_free_fill_size);
    internal_memset(p, flags()->free_fill_byte, fill_size);
  }
  if (flags()->tag_in_free &&
      atomic_load_relaxed(&hwasan_allocator_tagging_enabled))
    TagMemoryAligned((uptr)p, size,
                     t ? t->GenerateRandomTag() : kFallbackFreeTag);
  if (t) {
    AllocatorCache *cache = GetAllocatorCache(&t->malloc_storage());
    allocator.Deallocate(cache, p);
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocator.Deallocate(cache, p);
  }
}

void *HwasanReallocate(StackTrace *stack, void *user_old_p, uptr new_size,
                     uptr alignment) {
  alignment = Max(alignment, kShadowAlignment);
  new_size = RoundUpTo(new_size, kShadowAlignment);

  void *old_p = GetAddressFromPointer(user_old_p);
  Metadata *meta = reinterpret_cast<Metadata*>(allocator.GetMetaData(old_p));
  uptr old_size = meta->requested_size;
  uptr actually_allocated_size = allocator.GetActuallyAllocatedSize(old_p);
  if (new_size <= actually_allocated_size) {
    // We are not reallocating here.
    // FIXME: update stack trace for the allocation?
    meta->requested_size = new_size;
    if (!atomic_load_relaxed(&hwasan_allocator_tagging_enabled))
      return user_old_p;
    if (flags()->retag_in_realloc) {
      HwasanThread *t = GetCurrentThread();
      return (void *)TagMemoryAligned(
          (uptr)old_p, new_size,
          t ? t->GenerateRandomTag() : kFallbackAllocTag);
    }
    if (new_size > old_size) {
      tag_t tag = GetTagFromPointer((uptr)user_old_p);
      TagMemoryAligned((uptr)old_p + old_size, new_size - old_size, tag);
    }
    return user_old_p;
  }
  uptr memcpy_size = Min(new_size, old_size);
  void *new_p = HwasanAllocate(stack, new_size, alignment, false /*zeroise*/);
  if (new_p) {
    internal_memcpy(new_p, old_p, memcpy_size);
    HwasanDeallocate(stack, old_p);
  }
  return new_p;
}

void *HwasanCalloc(StackTrace *stack, uptr nmemb, uptr size) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportCallocOverflow(nmemb, size, stack);
  }
  return HwasanAllocate(stack, nmemb * size, sizeof(u64), true);
}

HwasanChunkView FindHeapChunkByAddress(uptr address) {
  void *block = allocator.GetBlockBegin(reinterpret_cast<void*>(address));
  if (!block)
    return HwasanChunkView();
  Metadata *metadata =
      reinterpret_cast<Metadata*>(allocator.GetMetaData(block));
  return HwasanChunkView(reinterpret_cast<uptr>(block), metadata);
}

static uptr AllocationSize(const void *user_ptr) {
  const void *p = GetAddressFromPointer(user_ptr);
  if (!p) return 0;
  const void *beg = allocator.GetBlockBegin(p);
  if (beg != p) return 0;
  Metadata *b = (Metadata *)allocator.GetMetaData(p);
  return b->requested_size;
}

void *hwasan_malloc(uptr size, StackTrace *stack) {
  return SetErrnoOnNull(HwasanAllocate(stack, size, sizeof(u64), false));
}

void *hwasan_calloc(uptr nmemb, uptr size, StackTrace *stack) {
  return SetErrnoOnNull(HwasanCalloc(stack, nmemb, size));
}

void *hwasan_realloc(void *ptr, uptr size, StackTrace *stack) {
  if (!ptr)
    return SetErrnoOnNull(HwasanAllocate(stack, size, sizeof(u64), false));
  if (size == 0) {
    HwasanDeallocate(stack, ptr);
    return nullptr;
  }
  return SetErrnoOnNull(HwasanReallocate(stack, ptr, size, sizeof(u64)));
}

void *hwasan_valloc(uptr size, StackTrace *stack) {
  return SetErrnoOnNull(
      HwasanAllocate(stack, size, GetPageSizeCached(), false));
}

void *hwasan_pvalloc(uptr size, StackTrace *stack) {
  uptr PageSize = GetPageSizeCached();
  if (UNLIKELY(CheckForPvallocOverflow(size, PageSize))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportPvallocOverflow(size, stack);
  }
  // pvalloc(0) should allocate one page.
  size = size ? RoundUpTo(size, PageSize) : PageSize;
  return SetErrnoOnNull(HwasanAllocate(stack, size, PageSize, false));
}

void *hwasan_aligned_alloc(uptr alignment, uptr size, StackTrace *stack) {
  if (UNLIKELY(!CheckAlignedAllocAlignmentAndSize(alignment, size))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAlignedAllocAlignment(size, alignment, stack);
  }
  return SetErrnoOnNull(HwasanAllocate(stack, size, alignment, false));
}

void *hwasan_memalign(uptr alignment, uptr size, StackTrace *stack) {
  if (UNLIKELY(!IsPowerOfTwo(alignment))) {
    errno = errno_EINVAL;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportInvalidAllocationAlignment(alignment, stack);
  }
  return SetErrnoOnNull(HwasanAllocate(stack, size, alignment, false));
}

int hwasan_posix_memalign(void **memptr, uptr alignment, uptr size,
                        StackTrace *stack) {
  if (UNLIKELY(!CheckPosixMemalignAlignment(alignment))) {
    if (AllocatorMayReturnNull())
      return errno_EINVAL;
    ReportInvalidPosixMemalignAlignment(alignment, stack);
  }
  void *ptr = HwasanAllocate(stack, size, alignment, false);
  if (UNLIKELY(!ptr))
    // OOM error is already taken care of by HwasanAllocate.
    return errno_ENOMEM;
  CHECK(IsAligned((uptr)ptr, alignment));
  *memptr = ptr;
  return 0;
}

}  // namespace __hwasan

using namespace __hwasan;

void __hwasan_enable_allocator_tagging() {
  atomic_store_relaxed(&hwasan_allocator_tagging_enabled, 1);
}

void __hwasan_disable_allocator_tagging() {
  atomic_store_relaxed(&hwasan_allocator_tagging_enabled, 0);
}

uptr __sanitizer_get_current_allocated_bytes() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatAllocated];
}

uptr __sanitizer_get_heap_size() {
  uptr stats[AllocatorStatCount];
  allocator.GetStats(stats);
  return stats[AllocatorStatMapped];
}

uptr __sanitizer_get_free_bytes() { return 1; }

uptr __sanitizer_get_unmapped_bytes() { return 1; }

uptr __sanitizer_get_estimated_allocated_size(uptr size) { return size; }

int __sanitizer_get_ownership(const void *p) { return AllocationSize(p) != 0; }

uptr __sanitizer_get_allocated_size(const void *p) { return AllocationSize(p); }
