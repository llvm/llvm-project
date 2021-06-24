//===-- dfsan_platform.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
// Platform specific information for DFSan.
//===----------------------------------------------------------------------===//

#ifndef DFSAN_PLATFORM_H
#define DFSAN_PLATFORM_H

#include "sanitizer_common/sanitizer_common.h"

namespace __dfsan {

using __sanitizer::uptr;

struct Mapping {
  static const uptr kShadowAddr = 0x100000008000;
  static const uptr kOriginAddr = 0x200000008000;
  static const uptr kUnusedAddr = 0x300000000000;
  static const uptr kAppAddr = 0x700000008000;
  static const uptr kShadowMask = ~0x600000000000;
};

enum MappingType {
  MAPPING_SHADOW_ADDR,
  MAPPING_ORIGIN_ADDR,
  MAPPING_UNUSED_ADDR,
  MAPPING_APP_ADDR,
  MAPPING_SHADOW_MASK
};

template<typename Mapping, int Type>
uptr MappingImpl(void) {
  switch (Type) {
    case MAPPING_SHADOW_ADDR: return Mapping::kShadowAddr;
#if defined(__x86_64__)
    case MAPPING_ORIGIN_ADDR:
      return Mapping::kOriginAddr;
#endif
    case MAPPING_UNUSED_ADDR:
      return Mapping::kUnusedAddr;
    case MAPPING_APP_ADDR: return Mapping::kAppAddr;
    case MAPPING_SHADOW_MASK: return Mapping::kShadowMask;
  }
}

template<int Type>
uptr MappingArchImpl(void) {
  return MappingImpl<Mapping, Type>();
}

ALWAYS_INLINE
uptr ShadowAddr() {
  return MappingArchImpl<MAPPING_SHADOW_ADDR>();
}

ALWAYS_INLINE
uptr OriginAddr() {
  return MappingArchImpl<MAPPING_ORIGIN_ADDR>();
}

ALWAYS_INLINE
uptr UnusedAddr() { return MappingArchImpl<MAPPING_UNUSED_ADDR>(); }

ALWAYS_INLINE
uptr AppAddr() {
  return MappingArchImpl<MAPPING_APP_ADDR>();
}

ALWAYS_INLINE
uptr ShadowMask() {
  return MappingArchImpl<MAPPING_SHADOW_MASK>();
}

}  // namespace __dfsan

#endif
