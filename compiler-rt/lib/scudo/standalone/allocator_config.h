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
//   // SizeClasMmap to use with the Primary.
//   using SizeClassMap = DefaultSizeClassMap;
//   // Indicates possible support for Memory Tagging.
//   static const bool MaySupportMemoryTagging = false;
//   // Defines the Primary allocator to use.
//   typedef SizeClassAllocator64<ExampleConfig> Primary;
//   // Log2 of the size of a size class region, as used by the Primary.
//   static const uptr PrimaryRegionSizeLog = 30U;
//   // Defines the type and scale of a compact pointer. A compact pointer can
//   // be understood as the offset of a pointer within the region it belongs
//   // to, in increments of a power-of-2 scale.
//   // eg: Ptr = Base + (CompactPtr << Scale).
//   typedef u32 PrimaryCompactPtrT;
//   static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
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
  static const bool MaySupportMemoryTagging = false;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 32U;
  typedef uptr PrimaryCompactPtrT;
  static const uptr PrimaryCompactPtrScale = 0;
#else
  typedef SizeClassAllocator32<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 19U;
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
#else
  typedef SizeClassAllocator32<AndroidConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 18U;
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
#else
  typedef SizeClassAllocator32<AndroidSvelteConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 16U;
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
  using SizeClassMap = DefaultSizeClassMap;
  static const bool MaySupportMemoryTagging = false;

  typedef SizeClassAllocator64<FuchsiaConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 30U;
  typedef u32 PrimaryCompactPtrT;
  static const uptr PrimaryCompactPtrScale = SCUDO_MIN_ALIGNMENT_LOG;
  static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;

  typedef MapAllocatorNoCache SecondaryCache;
  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 8U, 4U>; // Shared, max 8 TSDs.
};
#endif

#if SCUDO_ANDROID
typedef AndroidConfig Config;
#elif SCUDO_FUCHSIA
typedef FuchsiaConfig Config;
#else
typedef DefaultConfig Config;
#endif

} // namespace scudo

#endif // SCUDO_ALLOCATOR_CONFIG_H_
