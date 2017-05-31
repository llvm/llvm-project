//===-- msan_allocator.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of MemorySanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef MSAN_ALLOCATOR_H
#define MSAN_ALLOCATOR_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_allocator_interface.h"

namespace __msan {

struct Metadata {
  uptr requested_size;
};

struct MsanMapUnmapCallback {
  void OnMap(uptr p, uptr size) const {}
  void OnUnmap(uptr p, uptr size) const {
    __msan_unpoison((void *)p, size);

    // We are about to unmap a chunk of user memory.
    // Mark the corresponding shadow memory as not needed.
    uptr shadow_p = MEM_TO_SHADOW(p);
    ReleaseMemoryPagesToOS(shadow_p, shadow_p + size);
    if (__msan_get_track_origins()) {
      uptr origin_p = MEM_TO_ORIGIN(p);
      ReleaseMemoryPagesToOS(origin_p, origin_p + size);
    }
  }
};

#if defined(__mips64)
  static const uptr kMaxAllowedMallocSize = 2UL << 30;
  static const uptr kRegionSizeLog = 20;
  static const uptr kNumRegions = SANITIZER_MMAP_RANGE_SIZE >> kRegionSizeLog;
  typedef TwoLevelByteMap<(kNumRegions >> 12), 1 << 12> ByteMap;

  struct AP32 {
    static const uptr kSpaceBeg = 0;
    static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
    static const uptr kMetadataSize = sizeof(Metadata);
    typedef __sanitizer::CompactSizeClassMap SizeClassMap;
    static const uptr kRegionSizeLog = __msan::kRegionSizeLog;
    typedef __msan::ByteMap ByteMap;
    typedef MsanMapUnmapCallback MapUnmapCallback;
    static const uptr kFlags = 0;
  };
  typedef SizeClassAllocator32<AP32> PrimaryAllocator;
#elif defined(__x86_64__)
#if SANITIZER_LINUX && !defined(MSAN_LINUX_X86_64_OLD_MAPPING)
  static const uptr kAllocatorSpace = 0x700000000000ULL;
#else
  static const uptr kAllocatorSpace = 0x600000000000ULL;
#endif
  static const uptr kMaxAllowedMallocSize = 8UL << 30;

  struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
    static const uptr kSpaceBeg = kAllocatorSpace;
    static const uptr kSpaceSize = 0x40000000000; // 4T.
    static const uptr kMetadataSize = sizeof(Metadata);
    typedef DefaultSizeClassMap SizeClassMap;
    typedef MsanMapUnmapCallback MapUnmapCallback;
    static const uptr kFlags = 0;
  };

  typedef SizeClassAllocator64<AP64> PrimaryAllocator;

#elif defined(__powerpc64__)
  static const uptr kMaxAllowedMallocSize = 2UL << 30;  // 2G

  struct AP64 {  // Allocator64 parameters. Deliberately using a short name.
    static const uptr kSpaceBeg = 0x300000000000;
    static const uptr kSpaceSize = 0x020000000000; // 2T.
    static const uptr kMetadataSize = sizeof(Metadata);
    typedef DefaultSizeClassMap SizeClassMap;
    typedef MsanMapUnmapCallback MapUnmapCallback;
    static const uptr kFlags = 0;
  };

  typedef SizeClassAllocator64<AP64> PrimaryAllocator;
#elif defined(__aarch64__)
  static const uptr kMaxAllowedMallocSize = 2UL << 30;  // 2G
  static const uptr kRegionSizeLog = 20;
  static const uptr kNumRegions = SANITIZER_MMAP_RANGE_SIZE >> kRegionSizeLog;
  typedef TwoLevelByteMap<(kNumRegions >> 12), 1 << 12> ByteMap;

  struct AP32 {
    static const uptr kSpaceBeg = 0;
    static const u64 kSpaceSize = SANITIZER_MMAP_RANGE_SIZE;
    static const uptr kMetadataSize = sizeof(Metadata);
    typedef __sanitizer::CompactSizeClassMap SizeClassMap;
    static const uptr kRegionSizeLog = __msan::kRegionSizeLog;
    typedef __msan::ByteMap ByteMap;
    typedef MsanMapUnmapCallback MapUnmapCallback;
    static const uptr kFlags = 0;
  };
  typedef SizeClassAllocator32<AP32> PrimaryAllocator;
#endif
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<MsanMapUnmapCallback> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
                          SecondaryAllocator> Allocator;


Allocator &get_allocator();

struct MsanThreadLocalMallocStorage {
  uptr quarantine_cache[16];
  // Allocator cache contains atomic_uint64_t which must be 8-byte aligned.
  ALIGNED(8) uptr allocator_cache[96 * (512 * 8 + 16)];  // Opaque.
  void CommitBack();

 private:
  // These objects are allocated via mmap() and are zero-initialized.
  MsanThreadLocalMallocStorage() {}
};

} // namespace __msan
#endif // MSAN_ALLOCATOR_H
