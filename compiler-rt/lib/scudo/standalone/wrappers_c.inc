//===-- wrappers_c.inc ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_PREFIX
#error "Define SCUDO_PREFIX prior to including this file!"
#endif

// malloc-type functions have to be aligned to std::max_align_t. This is
// distinct from (1U << SCUDO_MIN_ALIGNMENT_LOG), since C++ new-type functions
// do not have to abide by the same requirement.
#ifndef SCUDO_MALLOC_ALIGNMENT
#define SCUDO_MALLOC_ALIGNMENT FIRST_32_SECOND_64(8U, 16U)
#endif

extern "C" {

INTERFACE WEAK void *SCUDO_PREFIX(calloc)(size_t nmemb, size_t size) {
  scudo::uptr Product;
  if (UNLIKELY(scudo::checkForCallocOverflow(size, nmemb, &Product))) {
    if (SCUDO_ALLOCATOR.canReturnNull()) {
      errno = ENOMEM;
      return nullptr;
    }
    scudo::reportCallocOverflow(nmemb, size);
  }
  return scudo::setErrnoOnNull(SCUDO_ALLOCATOR.allocate(
      Product, scudo::Chunk::Origin::Malloc, SCUDO_MALLOC_ALIGNMENT, true));
}

INTERFACE WEAK void SCUDO_PREFIX(free)(void *ptr) {
  SCUDO_ALLOCATOR.deallocate(ptr, scudo::Chunk::Origin::Malloc);
}

INTERFACE WEAK struct SCUDO_MALLINFO SCUDO_PREFIX(mallinfo)(void) {
  struct SCUDO_MALLINFO Info = {};
  scudo::StatCounters Stats;
  SCUDO_ALLOCATOR.getStats(Stats);
  // Space allocated in mmapped regions (bytes)
  Info.hblkhd = static_cast<__scudo_mallinfo_data_t>(Stats[scudo::StatMapped]);
  // Maximum total allocated space (bytes)
  Info.usmblks = Info.hblkhd;
  // Space in freed fastbin blocks (bytes)
  Info.fsmblks = static_cast<__scudo_mallinfo_data_t>(Stats[scudo::StatFree]);
  // Total allocated space (bytes)
  Info.uordblks =
      static_cast<__scudo_mallinfo_data_t>(Stats[scudo::StatAllocated]);
  // Total free space (bytes)
  Info.fordblks = Info.fsmblks;
  return Info;
}

INTERFACE WEAK void *SCUDO_PREFIX(malloc)(size_t size) {
  return scudo::setErrnoOnNull(SCUDO_ALLOCATOR.allocate(
      size, scudo::Chunk::Origin::Malloc, SCUDO_MALLOC_ALIGNMENT));
}

#if SCUDO_ANDROID
INTERFACE WEAK size_t SCUDO_PREFIX(malloc_usable_size)(const void *ptr) {
#else
INTERFACE WEAK size_t SCUDO_PREFIX(malloc_usable_size)(void *ptr) {
#endif
  return SCUDO_ALLOCATOR.getUsableSize(ptr);
}

INTERFACE WEAK void *SCUDO_PREFIX(memalign)(size_t alignment, size_t size) {
  // Android rounds up the alignment to a power of two if it isn't one.
  if (SCUDO_ANDROID) {
    if (UNLIKELY(!alignment)) {
      alignment = 1U;
    } else {
      if (UNLIKELY(!scudo::isPowerOfTwo(alignment)))
        alignment = scudo::roundUpToPowerOfTwo(alignment);
    }
  } else {
    if (UNLIKELY(!scudo::isPowerOfTwo(alignment))) {
      if (SCUDO_ALLOCATOR.canReturnNull()) {
        errno = EINVAL;
        return nullptr;
      }
      scudo::reportAlignmentNotPowerOfTwo(alignment);
    }
  }
  return SCUDO_ALLOCATOR.allocate(size, scudo::Chunk::Origin::Memalign,
                                  alignment);
}

INTERFACE WEAK int SCUDO_PREFIX(posix_memalign)(void **memptr, size_t alignment,
                                                size_t size) {
  if (UNLIKELY(scudo::checkPosixMemalignAlignment(alignment))) {
    if (!SCUDO_ALLOCATOR.canReturnNull())
      scudo::reportInvalidPosixMemalignAlignment(alignment);
    return EINVAL;
  }
  void *Ptr =
      SCUDO_ALLOCATOR.allocate(size, scudo::Chunk::Origin::Memalign, alignment);
  if (UNLIKELY(!Ptr))
    return ENOMEM;
  *memptr = Ptr;
  return 0;
}

INTERFACE WEAK void *SCUDO_PREFIX(pvalloc)(size_t size) {
  const scudo::uptr PageSize = scudo::getPageSizeCached();
  if (UNLIKELY(scudo::checkForPvallocOverflow(size, PageSize))) {
    if (SCUDO_ALLOCATOR.canReturnNull()) {
      errno = ENOMEM;
      return nullptr;
    }
    scudo::reportPvallocOverflow(size);
  }
  // pvalloc(0) should allocate one page.
  return scudo::setErrnoOnNull(SCUDO_ALLOCATOR.allocate(
      size ? scudo::roundUpTo(size, PageSize) : PageSize,
      scudo::Chunk::Origin::Memalign, PageSize));
}

INTERFACE WEAK void *SCUDO_PREFIX(realloc)(void *ptr, size_t size) {
  if (!ptr)
    return scudo::setErrnoOnNull(SCUDO_ALLOCATOR.allocate(
        size, scudo::Chunk::Origin::Malloc, SCUDO_MALLOC_ALIGNMENT));
  if (size == 0) {
    SCUDO_ALLOCATOR.deallocate(ptr, scudo::Chunk::Origin::Malloc);
    return nullptr;
  }
  return scudo::setErrnoOnNull(
      SCUDO_ALLOCATOR.reallocate(ptr, size, SCUDO_MALLOC_ALIGNMENT));
}

INTERFACE WEAK void *SCUDO_PREFIX(valloc)(size_t size) {
  return scudo::setErrnoOnNull(SCUDO_ALLOCATOR.allocate(
      size, scudo::Chunk::Origin::Memalign, scudo::getPageSizeCached()));
}

INTERFACE WEAK int SCUDO_PREFIX(malloc_iterate)(
    uintptr_t base, size_t size,
    void (*callback)(uintptr_t base, size_t size, void *arg), void *arg) {
  SCUDO_ALLOCATOR.iterateOverChunks(base, size, callback, arg);
  return 0;
}

INTERFACE WEAK void SCUDO_PREFIX(malloc_enable)() { SCUDO_ALLOCATOR.enable(); }

INTERFACE WEAK void SCUDO_PREFIX(malloc_disable)() {
  SCUDO_ALLOCATOR.disable();
}

void SCUDO_PREFIX(malloc_postinit)() {
  SCUDO_ALLOCATOR.initGwpAsan();
  pthread_atfork(SCUDO_PREFIX(malloc_disable), SCUDO_PREFIX(malloc_enable),
                 SCUDO_PREFIX(malloc_enable));
}

INTERFACE WEAK int SCUDO_PREFIX(mallopt)(int param, int value) {
  if (param == M_DECAY_TIME) {
    if (SCUDO_ANDROID) {
      if (value == 0) {
        // Will set the release values to their minimum values.
        value = INT32_MIN;
      } else {
        // Will set the release values to their maximum values.
        value = INT32_MAX;
      }
    }

    SCUDO_ALLOCATOR.setOption(scudo::Option::ReleaseInterval,
                              static_cast<scudo::sptr>(value));
    return 1;
  } else if (param == M_PURGE) {
    SCUDO_ALLOCATOR.releaseToOS();
    return 1;
  } else {
    scudo::Option option;
    switch (param) {
    case M_MEMTAG_TUNING:
      option = scudo::Option::MemtagTuning;
      break;
    case M_THREAD_DISABLE_MEM_INIT:
      option = scudo::Option::ThreadDisableMemInit;
      break;
    case M_CACHE_COUNT_MAX:
      option = scudo::Option::MaxCacheEntriesCount;
      break;
    case M_CACHE_SIZE_MAX:
      option = scudo::Option::MaxCacheEntrySize;
      break;
    case M_TSDS_COUNT_MAX:
      option = scudo::Option::MaxTSDsCount;
      break;
    default:
      return 0;
    }
    return SCUDO_ALLOCATOR.setOption(option, static_cast<scudo::sptr>(value));
  }
}

INTERFACE WEAK void *SCUDO_PREFIX(aligned_alloc)(size_t alignment,
                                                 size_t size) {
  if (UNLIKELY(scudo::checkAlignedAllocAlignmentAndSize(alignment, size))) {
    if (SCUDO_ALLOCATOR.canReturnNull()) {
      errno = EINVAL;
      return nullptr;
    }
    scudo::reportInvalidAlignedAllocAlignment(alignment, size);
  }
  return scudo::setErrnoOnNull(
      SCUDO_ALLOCATOR.allocate(size, scudo::Chunk::Origin::Malloc, alignment));
}

INTERFACE WEAK int SCUDO_PREFIX(malloc_info)(UNUSED int options, FILE *stream) {
  const scudo::uptr max_size =
      decltype(SCUDO_ALLOCATOR)::PrimaryT::SizeClassMap::MaxSize;
  auto *sizes = static_cast<scudo::uptr *>(
      SCUDO_PREFIX(calloc)(max_size, sizeof(scudo::uptr)));
  auto callback = [](uintptr_t, size_t size, void *arg) {
    auto *sizes = reinterpret_cast<scudo::uptr *>(arg);
    if (size < max_size)
      sizes[size]++;
  };
  SCUDO_ALLOCATOR.iterateOverChunks(0, -1ul, callback, sizes);

  fputs("<malloc version=\"scudo-1\">\n", stream);
  for (scudo::uptr i = 0; i != max_size; ++i)
    if (sizes[i])
      fprintf(stream, "<alloc size=\"%lu\" count=\"%lu\"/>\n", i, sizes[i]);
  fputs("</malloc>\n", stream);
  SCUDO_PREFIX(free)(sizes);
  return 0;
}

// Disable memory tagging for the heap. The caller must disable memory tag
// checks globally (e.g. by clearing TCF0 on aarch64) before calling this
// function, and may not re-enable them after calling the function. The program
// must be single threaded at the point when the function is called.
INTERFACE WEAK void SCUDO_PREFIX(malloc_disable_memory_tagging)() {
  SCUDO_ALLOCATOR.disableMemoryTagging();
}

// Sets whether scudo records stack traces and other metadata for allocations
// and deallocations. This function only has an effect if the allocator and
// hardware support memory tagging. The program must be single threaded at the
// point when the function is called.
INTERFACE WEAK void
SCUDO_PREFIX(malloc_set_track_allocation_stacks)(int track) {
  SCUDO_ALLOCATOR.setTrackAllocationStacks(track);
}

// Sets whether scudo zero-initializes all allocated memory. The program must
// be single threaded at the point when the function is called.
INTERFACE WEAK void SCUDO_PREFIX(malloc_set_zero_contents)(int zero_contents) {
  SCUDO_ALLOCATOR.setFillContents(zero_contents ? scudo::ZeroFill
                                                : scudo::NoFill);
}

// Sets whether scudo pattern-initializes all allocated memory. The program must
// be single threaded at the point when the function is called.
INTERFACE WEAK void
SCUDO_PREFIX(malloc_set_pattern_fill_contents)(int pattern_fill_contents) {
  SCUDO_ALLOCATOR.setFillContents(
      pattern_fill_contents ? scudo::PatternOrZeroFill : scudo::NoFill);
}

} // extern "C"
