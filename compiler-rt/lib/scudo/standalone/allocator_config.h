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
  static const bool MaySupportMemoryTagging = false;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 30U;
#else
  typedef SizeClassAllocator32<DefaultConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 19U;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = INT32_MIN;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = INT32_MAX;

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
  static const bool MaySupportMemoryTagging = true;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<AndroidConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 28U;
#else
  typedef SizeClassAllocator32<AndroidConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 18U;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = 1000;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = 1000;

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
  static const bool MaySupportMemoryTagging = false;

#if SCUDO_CAN_USE_PRIMARY64
  typedef SizeClassAllocator64<AndroidSvelteConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 27U;
#else
  typedef SizeClassAllocator32<AndroidSvelteConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 16U;
#endif
  static const s32 PrimaryMinReleaseToOsIntervalMs = 1000;
  static const s32 PrimaryMaxReleaseToOsIntervalMs = 1000;

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
  using SizeClassMap = DefaultSizeClassMap;
  static const bool MaySupportMemoryTagging = false;

  typedef SizeClassAllocator64<FuchsiaConfig> Primary;
  static const uptr PrimaryRegionSizeLog = 30U;
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
