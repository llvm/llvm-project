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

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_errno.h"
#include "sanitizer_common/sanitizer_stackdepot.h"
#include "hwasan.h"
#include "hwasan_allocator.h"
#include "hwasan_mapping.h"
#include "hwasan_thread.h"
#include "hwasan_report.h"

namespace __hwasan {

bool HwasanChunkView::IsAllocated() const {
  return metadata_ && metadata_->alloc_context_id && metadata_->requested_size;
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

static Allocator allocator;
static AllocatorCache fallback_allocator_cache;
static SpinMutex fallback_mutex;
static atomic_uint8_t hwasan_allocator_tagging_enabled;

static const tag_t kFallbackAllocTag = 0xBB;
static const tag_t kFallbackFreeTag = 0xBC;

void GetAllocatorStats(AllocatorStatCounters s) {
  allocator.GetStats(s);
}

void HwasanAllocatorInit() {
  atomic_store_relaxed(&hwasan_allocator_tagging_enabled,
                       !flags()->disable_allocator_tagging);
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator.Init(common_flags()->allocator_release_to_os_interval_ms);
}

void AllocatorSwallowThreadLocalCache(AllocatorCache *cache) {
  allocator.SwallowCache(cache);
}

static uptr TaggedSize(uptr size) {
  if (!size) size = 1;
  return RoundUpTo(size, kShadowAlignment);
}

static void *HwasanAllocate(StackTrace *stack, uptr orig_size, uptr alignment,
                            bool zeroise) {
  alignment = Max(alignment, kShadowAlignment);
  uptr size = TaggedSize(orig_size);

  if (size > kMaxAllowedMallocSize) {
    if (AllocatorMayReturnNull()) {
      Report("WARNING: HWAddressSanitizer failed to allocate 0x%zx bytes\n",
             size);
      return nullptr;
    }
    ReportAllocationSizeTooBig(size, kMaxAllowedMallocSize, stack);
  }
  Thread *t = GetCurrentThread();
  void *allocated;
  if (t) {
    allocated = allocator.Allocate(t->allocator_cache(), size, alignment);
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
  meta->requested_size = static_cast<u32>(orig_size);
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

static bool PointerAndMemoryTagsMatch(void *tagged_ptr) {
  CHECK(tagged_ptr);
  tag_t ptr_tag = GetTagFromPointer(reinterpret_cast<uptr>(tagged_ptr));
  tag_t mem_tag = *reinterpret_cast<tag_t *>(
      MemToShadow(reinterpret_cast<uptr>(UntagPtr(tagged_ptr))));
  return ptr_tag == mem_tag;
}

void HwasanDeallocate(StackTrace *stack, void *tagged_ptr) {
  CHECK(tagged_ptr);
  HWASAN_FREE_HOOK(tagged_ptr);

  if (!PointerAndMemoryTagsMatch(tagged_ptr))
    ReportInvalidFree(stack, reinterpret_cast<uptr>(tagged_ptr));

  void *untagged_ptr = UntagPtr(tagged_ptr);
  Metadata *meta =
      reinterpret_cast<Metadata *>(allocator.GetMetaData(untagged_ptr));
  uptr orig_size = meta->requested_size;
  u32 free_context_id = StackDepotPut(*stack);
  u32 alloc_context_id = meta->alloc_context_id;
  meta->requested_size = 0;
  meta->alloc_context_id = 0;
  // This memory will not be reused by anyone else, so we are free to keep it
  // poisoned.
  Thread *t = GetCurrentThread();
  if (flags()->max_free_fill_size > 0) {
    uptr fill_size = Min(orig_size, (uptr)flags()->max_free_fill_size);
    internal_memset(untagged_ptr, flags()->free_fill_byte, fill_size);
  }
  if (flags()->tag_in_free &&
      atomic_load_relaxed(&hwasan_allocator_tagging_enabled))
    TagMemoryAligned((uptr)untagged_ptr, TaggedSize(orig_size),
                     t ? t->GenerateRandomTag() : kFallbackFreeTag);
  if (t) {
    allocator.Deallocate(t->allocator_cache(), untagged_ptr);
    if (auto *ha = t->heap_allocations())
      ha->push({reinterpret_cast<uptr>(tagged_ptr), alloc_context_id,
                free_context_id, static_cast<u32>(orig_size)});
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocator.Deallocate(cache, untagged_ptr);
  }
}

void *HwasanReallocate(StackTrace *stack, void *tagged_ptr_old, uptr new_size,
                     uptr alignment) {
  if (!PointerAndMemoryTagsMatch(tagged_ptr_old))
    ReportInvalidFree(stack, reinterpret_cast<uptr>(tagged_ptr_old));

  void *tagged_ptr_new =
      HwasanAllocate(stack, new_size, alignment, false /*zeroise*/);
  if (tagged_ptr_old && tagged_ptr_new) {
    void *untagged_ptr_old =  UntagPtr(tagged_ptr_old);
    Metadata *meta =
        reinterpret_cast<Metadata *>(allocator.GetMetaData(untagged_ptr_old));
    internal_memcpy(UntagPtr(tagged_ptr_new), untagged_ptr_old,
                    Min(new_size, static_cast<uptr>(meta->requested_size)));
    HwasanDeallocate(stack, tagged_ptr_old);
  }
  return tagged_ptr_new;
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

static uptr AllocationSize(const void *tagged_ptr) {
  const void *untagged_ptr = UntagPtr(tagged_ptr);
  if (!untagged_ptr) return 0;
  const void *beg = allocator.GetBlockBegin(untagged_ptr);
  if (beg != untagged_ptr) return 0;
  Metadata *b = (Metadata *)allocator.GetMetaData(untagged_ptr);
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
