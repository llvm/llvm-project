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

// Default configurations for various platforms.

struct DefaultConfig {
  using SizeClassMap = DefaultSizeClassMap;
#if SCUDO_CAN_USE_PRIMARY64
  // 1GB Regions
  typedef SizeClassAllocator64<SizeClassMap, 30U> Primary;
#else
  // 512KB regions
  typedef SizeClassAllocator32<SizeClassMap, 19U> Primary;
#endif
  typedef MapAllocatorCache<DefaultConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 32U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 32U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 1UL << 19;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = INT32_MAX;

  template <class A> using TSDRegistryT = TSDRegistryExT<A>; // Exclusive
};

struct AndroidConfig {
  using SizeClassMap = AndroidSizeClassMap;
#if SCUDO_CAN_USE_PRIMARY64
  // 256MB regions
  typedef SizeClassAllocator64<SizeClassMap, 28U, 1000, 1000,
                               /*MaySupportMemoryTagging=*/true>
      Primary;
#else
  // 256KB regions
  typedef SizeClassAllocator32<SizeClassMap, 18U, 1000, 1000> Primary;
#endif
  typedef MapAllocatorCache<AndroidConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 256U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 32U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 2UL << 20;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = 0;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = 1000;

  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 8U, 2U>; // Shared, max 8 TSDs.
};

struct AndroidSvelteConfig {
  using SizeClassMap = SvelteSizeClassMap;
#if SCUDO_CAN_USE_PRIMARY64
  // 128MB regions
  typedef SizeClassAllocator64<SizeClassMap, 27U, 1000, 1000> Primary;
#else
  // 64KB regions
  typedef SizeClassAllocator32<SizeClassMap, 16U, 1000, 1000> Primary;
#endif
  typedef MapAllocatorCache<AndroidSvelteConfig> SecondaryCache;
  static const u32 SecondaryCacheEntriesArraySize = 16U;
  static const u32 SecondaryCacheDefaultMaxEntriesCount = 4U;
  static const uptr SecondaryCacheDefaultMaxEntrySize = 1UL << 18;
  static const s32 SecondaryCacheMinReleaseToOsIntervalMs = 0;
  static const s32 SecondaryCacheMaxReleaseToOsIntervalMs = 0;

  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 2U, 1U>; // Shared, max 2 TSDs.
};

#if SCUDO_CAN_USE_PRIMARY64
struct FuchsiaConfig {
  // 1GB Regions
  typedef SizeClassAllocator64<DefaultSizeClassMap, 30U> Primary;
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
