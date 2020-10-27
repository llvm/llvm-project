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
  typedef MapAllocator<MapAllocatorCache<>> Secondary;
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
  // Cache blocks up to 2MB
  typedef MapAllocator<MapAllocatorCache<256U, 32U, 2UL << 20, 0, 1000>>
      Secondary;
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
  typedef MapAllocator<MapAllocatorCache<16U, 4U, 1UL << 18, 0, 0>> Secondary;
  template <class A>
  using TSDRegistryT = TSDRegistrySharedT<A, 2U, 1U>; // Shared, max 2 TSDs.
};

#if SCUDO_CAN_USE_PRIMARY64
struct FuchsiaConfig {
  // 1GB Regions
  typedef SizeClassAllocator64<DefaultSizeClassMap, 30U> Primary;
  typedef MapAllocator<MapAllocatorNoCache> Secondary;
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
