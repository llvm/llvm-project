//===-- hwasan_allocator.cpp ------------------------ ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "hwasan_checks.h"
#include "hwasan_mapping.h"
#include "hwasan_malloc_bisect.h"
#include "hwasan_thread.h"
#include "hwasan_report.h"
#include "lsan/lsan_common.h"

namespace __hwasan {

static Allocator allocator;
static AllocatorCache fallback_allocator_cache;
static SpinMutex fallback_mutex;
static atomic_uint8_t hwasan_allocator_tagging_enabled;

static constexpr tag_t kFallbackAllocTag = 0xBB & kTagMask;
static constexpr tag_t kFallbackFreeTag = 0xBC;

enum {
  // Either just allocated by underlying allocator, but AsanChunk is not yet
  // ready, or almost returned to undelying allocator and AsanChunk is already
  // meaningless.
  CHUNK_INVALID = 0,
  // The chunk is allocated and not yet freed.
  CHUNK_ALLOCATED = 1,
};


// Initialized in HwasanAllocatorInit, an never changed.
static ALIGNED(16) u8 tail_magic[kShadowAlignment - 1];

bool HwasanChunkView::IsAllocated() const {
  return metadata_ && metadata_->IsAllocated();
}

uptr HwasanChunkView::Beg() const {
  return block_;
}
uptr HwasanChunkView::End() const {
  return Beg() + UsedSize();
}
uptr HwasanChunkView::UsedSize() const {
  return metadata_->GetRequestedSize();
}
u32 HwasanChunkView::GetAllocStackId() const {
  return metadata_->GetAllocStackId();
}

uptr HwasanChunkView::ActualSize() const {
  return allocator.GetActuallyAllocatedSize(reinterpret_cast<void *>(block_));
}

bool HwasanChunkView::FromSmallHeap() const {
  return allocator.FromPrimary(reinterpret_cast<void *>(block_));
}

bool HwasanChunkView::AddrIsInside(uptr addr) const {
  return (addr >= Beg()) && (addr < Beg() + UsedSize());
}

inline void Metadata::SetAllocated(u32 stack, u64 size) {
  Thread *t = GetCurrentThread();
  u64 context = t ? t->unique_id() : kMainTid;
  context <<= 32;
  context += stack;
  requested_size_low = size & ((1ul << 32) - 1);
  requested_size_high = size >> 32;
  atomic_store(&alloc_context_id, context, memory_order_relaxed);
  atomic_store(&chunk_state, CHUNK_ALLOCATED, memory_order_release);
}

inline void Metadata::SetUnallocated() {
  atomic_store(&chunk_state, CHUNK_INVALID, memory_order_release);
  requested_size_low = 0;
  requested_size_high = 0;
  atomic_store(&alloc_context_id, 0, memory_order_relaxed);
}

inline bool Metadata::IsAllocated() const {
  return atomic_load(&chunk_state, memory_order_relaxed) == CHUNK_ALLOCATED &&
         GetRequestedSize();
}

inline u64 Metadata::GetRequestedSize() const {
  return (static_cast<u64>(requested_size_high) << 32) + requested_size_low;
}

inline u32 Metadata::GetAllocStackId() const {
  return atomic_load(&alloc_context_id, memory_order_relaxed);
}

static const uptr kChunkHeaderSize = sizeof(HwasanChunkView);

void GetAllocatorStats(AllocatorStatCounters s) {
  allocator.GetStats(s);
}

inline void Metadata::SetLsanTag(__lsan::ChunkTag tag) {
  lsan_tag = tag;
}

inline __lsan::ChunkTag Metadata::GetLsanTag() const {
  return static_cast<__lsan::ChunkTag>(lsan_tag);
}

uptr GetAliasRegionStart() {
#if defined(HWASAN_ALIASING_MODE)
  constexpr uptr kAliasRegionOffset = 1ULL << (kTaggableRegionCheckShift - 1);
  uptr AliasRegionStart =
      __hwasan_shadow_memory_dynamic_address + kAliasRegionOffset;

  CHECK_EQ(AliasRegionStart >> kTaggableRegionCheckShift,
           __hwasan_shadow_memory_dynamic_address >> kTaggableRegionCheckShift);
  CHECK_EQ(
      (AliasRegionStart + kAliasRegionOffset - 1) >> kTaggableRegionCheckShift,
      __hwasan_shadow_memory_dynamic_address >> kTaggableRegionCheckShift);
  return AliasRegionStart;
#else
  return 0;
#endif
}

void HwasanAllocatorInit() {
  atomic_store_relaxed(&hwasan_allocator_tagging_enabled,
                       !flags()->disable_allocator_tagging);
  SetAllocatorMayReturnNull(common_flags()->allocator_may_return_null);
  allocator.Init(common_flags()->allocator_release_to_os_interval_ms,
                 GetAliasRegionStart());
  for (uptr i = 0; i < sizeof(tail_magic); i++)
    tail_magic[i] = GetCurrentThread()->GenerateRandomTag();
}

void HwasanAllocatorLock() { allocator.ForceLock(); }

void HwasanAllocatorUnlock() { allocator.ForceUnlock(); }

void AllocatorSwallowThreadLocalCache(AllocatorCache *cache) {
  allocator.SwallowCache(cache);
}

static uptr TaggedSize(uptr size) {
  if (!size) size = 1;
  uptr new_size = RoundUpTo(size, kShadowAlignment);
  CHECK_GE(new_size, size);
  return new_size;
}

static void *HwasanAllocate(StackTrace *stack, uptr orig_size, uptr alignment,
                            bool zeroise) {
  if (orig_size > kMaxAllowedMallocSize) {
    if (AllocatorMayReturnNull()) {
      Report("WARNING: HWAddressSanitizer failed to allocate 0x%zx bytes\n",
             orig_size);
      return nullptr;
    }
    ReportAllocationSizeTooBig(orig_size, kMaxAllowedMallocSize, stack);
  }
  if (UNLIKELY(IsRssLimitExceeded())) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportRssLimitExceeded(stack);
  }

  alignment = Max(alignment, kShadowAlignment);
  uptr size = TaggedSize(orig_size);
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
  if (zeroise) {
    internal_memset(allocated, 0, size);
  } else if (flags()->max_malloc_fill_size > 0) {
    uptr fill_size = Min(size, (uptr)flags()->max_malloc_fill_size);
    internal_memset(allocated, flags()->malloc_fill_byte, fill_size);
  }
  if (size != orig_size) {
    u8 *tail = reinterpret_cast<u8 *>(allocated) + orig_size;
    uptr tail_length = size - orig_size;
    internal_memcpy(tail, tail_magic, tail_length - 1);
    // Short granule is excluded from magic tail, so we explicitly untag.
    tail[tail_length - 1] = 0;
  }

  void *user_ptr = allocated;
  // Tagging can only be skipped when both tag_in_malloc and tag_in_free are
  // false. When tag_in_malloc = false and tag_in_free = true malloc needs to
  // retag to 0.
  if (InTaggableRegion(reinterpret_cast<uptr>(user_ptr)) &&
      (flags()->tag_in_malloc || flags()->tag_in_free) &&
      atomic_load_relaxed(&hwasan_allocator_tagging_enabled)) {
    if (flags()->tag_in_malloc && malloc_bisect(stack, orig_size)) {
      tag_t tag = t ? t->GenerateRandomTag() : kFallbackAllocTag;
      uptr tag_size = orig_size ? orig_size : 1;
      uptr full_granule_size = RoundDownTo(tag_size, kShadowAlignment);
      user_ptr =
          (void *)TagMemoryAligned((uptr)user_ptr, full_granule_size, tag);
      if (full_granule_size != tag_size) {
        u8 *short_granule =
            reinterpret_cast<u8 *>(allocated) + full_granule_size;
        TagMemoryAligned((uptr)short_granule, kShadowAlignment,
                         tag_size % kShadowAlignment);
        short_granule[kShadowAlignment - 1] = tag;
      }
    } else {
      user_ptr = (void *)TagMemoryAligned((uptr)user_ptr, size, 0);
    }
  }

  Metadata *meta =
      reinterpret_cast<Metadata *>(allocator.GetMetaData(allocated));
  meta->SetAllocated(StackDepotPut(*stack), orig_size);
  RunMallocHooks(user_ptr, size);
  return user_ptr;
}

static bool PointerAndMemoryTagsMatch(void *tagged_ptr) {
  CHECK(tagged_ptr);
  uptr tagged_uptr = reinterpret_cast<uptr>(tagged_ptr);
  if (!InTaggableRegion(tagged_uptr))
    return true;
  tag_t mem_tag = *reinterpret_cast<tag_t *>(
      MemToShadow(reinterpret_cast<uptr>(UntagPtr(tagged_ptr))));
  return PossiblyShortTagMatches(mem_tag, tagged_uptr, 1);
}

static bool CheckInvalidFree(StackTrace *stack, void *untagged_ptr,
                             void *tagged_ptr) {
  // This function can return true if halt_on_error is false.
  if (!MemIsApp(reinterpret_cast<uptr>(untagged_ptr)) ||
      !PointerAndMemoryTagsMatch(tagged_ptr)) {
    ReportInvalidFree(stack, reinterpret_cast<uptr>(tagged_ptr));
    return true;
  }
  return false;
}

static void HwasanDeallocate(StackTrace *stack, void *tagged_ptr) {
  CHECK(tagged_ptr);
  RunFreeHooks(tagged_ptr);

  bool in_taggable_region =
      InTaggableRegion(reinterpret_cast<uptr>(tagged_ptr));
  void *untagged_ptr = in_taggable_region ? UntagPtr(tagged_ptr) : tagged_ptr;

  if (CheckInvalidFree(stack, untagged_ptr, tagged_ptr))
    return;

  void *aligned_ptr = reinterpret_cast<void *>(
      RoundDownTo(reinterpret_cast<uptr>(untagged_ptr), kShadowAlignment));
  tag_t pointer_tag = GetTagFromPointer(reinterpret_cast<uptr>(tagged_ptr));
  Metadata *meta =
      reinterpret_cast<Metadata *>(allocator.GetMetaData(aligned_ptr));
  if (!meta) {
    ReportInvalidFree(stack, reinterpret_cast<uptr>(tagged_ptr));
    return;
  }
  uptr orig_size = meta->GetRequestedSize();
  u32 free_context_id = StackDepotPut(*stack);
  u32 alloc_context_id = meta->GetAllocStackId();

  // Check tail magic.
  uptr tagged_size = TaggedSize(orig_size);
  if (flags()->free_checks_tail_magic && orig_size &&
      tagged_size != orig_size) {
    uptr tail_size = tagged_size - orig_size - 1;
    CHECK_LT(tail_size, kShadowAlignment);
    void *tail_beg = reinterpret_cast<void *>(
        reinterpret_cast<uptr>(aligned_ptr) + orig_size);
    tag_t short_granule_memtag = *(reinterpret_cast<tag_t *>(
        reinterpret_cast<uptr>(tail_beg) + tail_size));
    if (tail_size &&
        (internal_memcmp(tail_beg, tail_magic, tail_size) ||
         (in_taggable_region && pointer_tag != short_granule_memtag)))
      ReportTailOverwritten(stack, reinterpret_cast<uptr>(tagged_ptr),
                            orig_size, tail_magic);
  }

  // TODO(kstoimenov): consider meta->SetUnallocated(free_context_id).
  meta->SetUnallocated();
  // This memory will not be reused by anyone else, so we are free to keep it
  // poisoned.
  Thread *t = GetCurrentThread();
  if (flags()->max_free_fill_size > 0) {
    uptr fill_size =
        Min(TaggedSize(orig_size), (uptr)flags()->max_free_fill_size);
    internal_memset(aligned_ptr, flags()->free_fill_byte, fill_size);
  }
  if (in_taggable_region && flags()->tag_in_free && malloc_bisect(stack, 0) &&
      atomic_load_relaxed(&hwasan_allocator_tagging_enabled)) {
    // Always store full 8-bit tags on free to maximize UAF detection.
    tag_t tag;
    if (t) {
      // Make sure we are not using a short granule tag as a poison tag. This
      // would make us attempt to read the memory on a UaF.
      // The tag can be zero if tagging is disabled on this thread.
      do {
        tag = t->GenerateRandomTag(/*num_bits=*/8);
      } while (
          UNLIKELY((tag < kShadowAlignment || tag == pointer_tag) && tag != 0));
    } else {
      static_assert(kFallbackFreeTag >= kShadowAlignment,
                    "fallback tag must not be a short granule tag.");
      tag = kFallbackFreeTag;
    }
    TagMemoryAligned(reinterpret_cast<uptr>(aligned_ptr), TaggedSize(orig_size),
                     tag);
  }
  if (t) {
    allocator.Deallocate(t->allocator_cache(), aligned_ptr);
    if (auto *ha = t->heap_allocations())
      ha->push({reinterpret_cast<uptr>(tagged_ptr), alloc_context_id,
                free_context_id, static_cast<u32>(orig_size)});
  } else {
    SpinMutexLock l(&fallback_mutex);
    AllocatorCache *cache = &fallback_allocator_cache;
    allocator.Deallocate(cache, aligned_ptr);
  }
}

static void *HwasanReallocate(StackTrace *stack, void *tagged_ptr_old,
                              uptr new_size, uptr alignment) {
  void *untagged_ptr_old =
      InTaggableRegion(reinterpret_cast<uptr>(tagged_ptr_old))
          ? UntagPtr(tagged_ptr_old)
          : tagged_ptr_old;
  if (CheckInvalidFree(stack, untagged_ptr_old, tagged_ptr_old))
    return nullptr;
  void *tagged_ptr_new =
      HwasanAllocate(stack, new_size, alignment, false /*zeroise*/);
  if (tagged_ptr_old && tagged_ptr_new) {
    Metadata *meta =
        reinterpret_cast<Metadata *>(allocator.GetMetaData(untagged_ptr_old));
    internal_memcpy(
        UntagPtr(tagged_ptr_new), untagged_ptr_old,
        Min(new_size, static_cast<uptr>(meta->GetRequestedSize())));
    HwasanDeallocate(stack, tagged_ptr_old);
  }
  return tagged_ptr_new;
}

static void *HwasanCalloc(StackTrace *stack, uptr nmemb, uptr size) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportCallocOverflow(nmemb, size, stack);
  }
  return HwasanAllocate(stack, nmemb * size, sizeof(u64), true);
}

HwasanChunkView FindHeapChunkByAddress(uptr address) {
  if (!allocator.PointerIsMine(reinterpret_cast<void *>(address)))
    return HwasanChunkView();
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
  Metadata *b = (Metadata *)allocator.GetMetaData(untagged_ptr);
  if (beg != untagged_ptr) return 0;
  return b->GetRequestedSize();
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

void *hwasan_reallocarray(void *ptr, uptr nmemb, uptr size, StackTrace *stack) {
  if (UNLIKELY(CheckForCallocOverflow(size, nmemb))) {
    errno = errno_ENOMEM;
    if (AllocatorMayReturnNull())
      return nullptr;
    ReportReallocArrayOverflow(nmemb, size, stack);
  }
  return hwasan_realloc(ptr, nmemb * size, stack);
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

void hwasan_free(void *ptr, StackTrace *stack) {
  return HwasanDeallocate(stack, ptr);
}

}  // namespace __hwasan

// --- Implementation of LSan-specific functions --- {{{1
namespace __lsan {

void LockAllocator() {
  __hwasan::HwasanAllocatorLock();
}

void UnlockAllocator() {
  __hwasan::HwasanAllocatorUnlock();
}

void GetAllocatorGlobalRange(uptr *begin, uptr *end) {
  *begin = (uptr)&__hwasan::allocator;
  *end = *begin + sizeof(__hwasan::allocator);
}

uptr PointsIntoChunk(void *p) {
  uptr addr = reinterpret_cast<uptr>(p);
  __hwasan::HwasanChunkView view = __hwasan::FindHeapChunkByAddress(addr);
  if (!view.IsAllocated()) 
    return 0;
  uptr chunk = view.Beg();
  if (view.AddrIsInside(addr))
    return chunk;
  if (IsSpecialCaseOfOperatorNew0(chunk, view.UsedSize(), addr))
    return chunk;
  return 0;
}

uptr GetUserBegin(uptr chunk) {
  return __hwasan::FindHeapChunkByAddress(chunk).Beg();
}

LsanMetadata::LsanMetadata(uptr chunk) {
  metadata_ = chunk ? reinterpret_cast<__hwasan::Metadata *>(
                          chunk - __hwasan::kChunkHeaderSize)
                    : nullptr;
}

bool LsanMetadata::allocated() const {
  if (!metadata_)
    return false;
  __hwasan::Metadata *m = reinterpret_cast<__hwasan::Metadata *>(metadata_);
  return m->IsAllocated();
}

ChunkTag LsanMetadata::tag() const {
  __hwasan::Metadata *m = reinterpret_cast<__hwasan::Metadata *>(metadata_);
  return m->GetLsanTag();
}

void LsanMetadata::set_tag(ChunkTag value) {
  __hwasan::Metadata *m = reinterpret_cast<__hwasan::Metadata *>(metadata_);
  m->SetLsanTag(value);
}

uptr LsanMetadata::requested_size() const {
  __hwasan::Metadata *m = reinterpret_cast<__hwasan::Metadata *>(metadata_);
  return m->GetRequestedSize();
}

u32 LsanMetadata::stack_trace_id() const {
  __hwasan::Metadata *m = reinterpret_cast<__hwasan::Metadata *>(metadata_);
  return m->GetAllocStackId();
}

void ForEachChunk(ForEachChunkCallback callback, void *arg) {
  __hwasan::allocator.ForEachChunk(callback, arg);
}

}  // namespace __lsan

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
