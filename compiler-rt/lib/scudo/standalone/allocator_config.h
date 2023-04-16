//===-- allocator_config.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ALLOCATOR_CONFIG_H_
#define SCUDO_ALLOCATOR_CONFIG_H_

#include "combined.h"
#include "common.h"
#include "flags.h"
#include "primary32.h"
#include "primary64.h"
#include "secondary.h"
#include "size_class_map.h"
#include "tsd_exclusive.h"
#include "tsd_shared.h"

namespace scudo {

// The combined allocator uses a structure as a template argument that
// specifies the configuration options for the various subcomponents of the
// allocator.
//
// struct ExampleConfig {
//   // SizeClassMap to use with the Primary.
//   using SizeClassMap = DefaultSizeClassMap;
//   // Indicates possible support for Memory Tagging.
//   static const bool MaySupportMemoryTagging = false;
//   // Defines the Primary allocator to use.
//   typedef SizeClassAllocator64<ExampleConfig> Primary;
//   // Log2 of the size of a size class region, as used by the Primary.
//   static const uptr PrimaryRegionSizeLog = 30U;
//   // Log2 of the size of block group, as used by the Primary. Each group
//   // contains a range of memory addresses, blocks in the range will belong to
//   // the same group. In general, single region may have 1 or 2MB group size.
//   // Multiple regions will have the group size equal to the region size
//   // because the region size is usually smaller than 1 MB.
//   // Smaller value gives fine-grained control of memory usage but the trade
//   // off is that it may take longer time of deallocation.
//   static const uptr PrimaryGroupSizeLog = 20U;
//   // Defines the type and scale of a compact pointer. A compact pointer can
//   // be understood as the offset of a pointer within the region it belongs
//   // to, in increments of a power-of-2 scale.
//   // eg: Ptr = Base + (CompactPtr << Scale).
//   typedef u32 PrimaryCompactPtrT;
//   static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
//   // Indicates support for offsetting the start of a region by
//   // a random number of pages. Only used with primary64.
//   static const bool PrimaryEnableRandomOffset = true;
//   // Call map for user memory with at least this size. Only used with
//   // primary64.
//   static const uptr PrimaryMapSizeIncrement = 1UL << 18;
//   // Defines the minimal & maximal release interval that can be set.
//   static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
//   static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;
//   // Defines the type of cache used by the Secondary. Some additional
//   // configuration entries can be necessary depending on the Cache.
//   typedef MapAllocatorNoCache SecondaryCache;
//   // Thread-Specific Data Registry used, shared or exclusive.
//   template <class A> using TSDRegistryT = TSDRegistrySharedT<A, 8U, 4U>;
// };

// Default configurations for various platforms.

struct DefaultConfig {
  using SizeClassMap = DefaultSizeClassMap;
  static const bool MaySupportMemoryTagging = true;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 32U;
  static const uptr PrimaryGroupSizeLog = 21U;
  typedef uptr PrimaryCompactPtrT;
  static const uptr PrimaryCompactPtrScale = 0;
  static const bool PrimaryEnableRandomOffset = true;
  static const uptr PrimaryMapSizeIncrement = 1UL << 18;
#else
  typedef SizeClassAllocator32<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 19U;
  static const uptr PrimaryGroupSizeLog = 19U;
  typedef uptr PrimaryCompactPtrT;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;

  typedef MapAllocatorCache<DefaultConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 32U;
  static const u32 SecondaryCacheQuarantineSize = 0U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 32U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 1UL << 19;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = INT32_MAX;

  template <class A> using TSDRegistryT = TSDRegistryExT<A>; // Exclusive
};
struct AndroidConfig {
  using SizeClassMap = AndroidSizeClassMap;
  static const bool MaySupportMemoryTagging = true;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<AndroidConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 28U;
  typedef u32 PrimaryCompactPtrT;
  static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
  static const uptr PrimaryGroupSizeLog = 20U;
  static const bool PrimaryEnableRandomOffset = true;
  static const uptr PrimaryMapSizeIncrement = 1UL << 18;
#else
  typedef SizeClassAllocator32<AndroidConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 18U;
  static const uptr PrimaryGroupSizeLog = 18U;
  typedef uptr PrimaryCompactPtrT;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = 1000;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = 1000;

  typedef MapAllocatorCache<AndroidConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 256U;
  static const u32 SecondaryCacheQuarantineSize = 32U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 32U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 2UL << 20;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = 0;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = 1000;

  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 8U, 2U>; // Shared, max 8 TSDs.
};

struct AndroidSvelteConfig {
  using SizeClassMap = SvelteSizeClassMap;
  static const bool MaySupportMemoryTagging = false;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<AndroidSvelteConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 27U;
  typedef u32 PrimaryCompactPtrT;
  static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
  static const uptr PrimaryGroupSizeLog = 18U;
  static const bool PrimaryEnableRandomOffset = true;
  static const uptr PrimaryMapSizeIncrement = 1UL << 18;
#else
  typedef SizeClassAllocator32<AndroidSvelteConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 16U;
  static const uptr PrimaryGroupSizeLog = 16U;
  typedef uptr PrimaryCompactPtrT;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = 1000;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = 1000;

  typedef MapAllocatorCache<AndroidSvelteConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 16U;
  static const u32 SecondaryCacheQuarantineSize = 32U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 4U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 1UL << 18;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = 0;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = 0;

  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 2U, 1U>; // Shared, max 2 TSDs.
};

#if SCUDO_CAN_USE_PRIMARY64
struct FuchsiaConfig {
  using SizeClassMap = FuchsiaSizeClassMap;
  static const bool MaySupportMemoryTagging = false;

  typedef SizeClassAllocator64<FuchsiaConfig> Primary;
// Support 39-bit VMA for riscv-64
#if SCUDO_RISCV64
  static const uptr PrimaryRegionSizeLog = 28U;
  static const uptr PrimaryGroupSizeLog = 19U;
#else
  static const uptr PrimaryRegionSizeLog = 30U;
  static const uptr PrimaryGroupSizeLog = 21U;
#endif
  typedef u32 PrimaryCompactPtrT;
  static const bool PrimaryEnableRandomOffset = true;
  static const uptr PrimaryMapSizeIncrement = 1UL << 18;
  static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
  static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;

  typedef MapAllocatorNoCache SecondaryCache;
  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 8U, 4U>; // Shared, max 8 TSDs.
};

struct TrustyConfig {
  using SizeClassMap = TrustySizeClassMap;
  static const bool MaySupportMemoryTagging = false;

  typedef SizeClassAllocator64<TrustyConfig> Primary;
  // Some apps have 1 page of heap total so small regions are necessary.
  static const uptr PrimaryRegionSizeLog = 10U;
  static const uptr PrimaryGroupSizeLog = 10U;
  typedef u32 PrimaryCompactPtrT;
  static const bool PrimaryEnableRandomOffset = false;
  // Trusty is extremely memory-constrained so minimally round up map calls.
  static const uptr PrimaryMapSizeIncrement = 1UL << 4;
  static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
  static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;

  typedef MapAllocatorNoCache SecondaryCache;
  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 1U, 1U>; // Shared, max 1 TSD.
};
#endif

#if SCUDO_ANDROID
typedef AndroidConfig Config;
#elif SCUDO_FUCHSIA
typedef FuchsiaConfig Config;
#elif SCUDO_TRUSTY
typedef TrustyConfig Config;
#else
typedef DefaultConfig Config;
#endif

} // namespace scudo

#endif // SCUDO_ALLOCATOR_CONFIG_H_
