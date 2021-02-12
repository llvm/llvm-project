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

namespace __dfsan {

#if defined(__x86_64__)
struct Mapping {
  static const uptr kShadowAddr = 0x10000;
  static const uptr kOriginAddr = 0x200000000000;
  static const uptr kUnionTableAddr = 0x300000000000;
  static const uptr kAppAddr = 0x700000008000;
  static const uptr kShadowMask = ~0x700000000000;
};
#elif defined(__mips64)
struct Mapping {
  static const uptr kShadowAddr = 0x10000;
  static const uptr kUnionTableAddr = 0x2000000000;
  static const uptr kAppAddr = 0xF000008000;
  static const uptr kShadowMask = ~0xF000000000;
};
#elif defined(__aarch64__)
struct Mapping39 {
  static const uptr kShadowAddr = 0x10000;
  static const uptr kUnionTableAddr = 0x1000000000;
  static const uptr kAppAddr = 0x7000008000;
  static const uptr kShadowMask = ~0x7800000000;
};

struct Mapping42 {
  static const uptr kShadowAddr = 0x10000;
  static const uptr kUnionTableAddr = 0x8000000000;
  static const uptr kAppAddr = 0x3ff00008000;
  static const uptr kShadowMask = ~0x3c000000000;
};

struct Mapping48 {
  static const uptr kShadowAddr = 0x10000;
  static const uptr kUnionTableAddr = 0x8000000000;
  static const uptr kAppAddr = 0xffff00008000;
  static const uptr kShadowMask = ~0xfffff0000000;
};

extern int vmaSize;
# define DFSAN_RUNTIME_VMA 1
#else
# error "DFSan not supported for this platform!"
#endif

enum MappingType {
  MAPPING_SHADOW_ADDR,
#if defined(__x86_64__)
  MAPPING_ORIGIN_ADDR,
#endif
  MAPPING_UNION_TABLE_ADDR,
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
    case MAPPING_UNION_TABLE_ADDR: return Mapping::kUnionTableAddr;
    case MAPPING_APP_ADDR: return Mapping::kAppAddr;
    case MAPPING_SHADOW_MASK: return Mapping::kShadowMask;
  }
}

template<int Type>
uptr MappingArchImpl(void) {
#ifdef __aarch64__
  switch (vmaSize) {
    case 39: return MappingImpl<Mapping39, Type>();
    case 42: return MappingImpl<Mapping42, Type>();
    case 48: return MappingImpl<Mapping48, Type>();
  }
  DCHECK(0);
  return 0;
#else
  return MappingImpl<Mapping, Type>();
#endif
}

ALWAYS_INLINE
uptr ShadowAddr() {
  return MappingArchImpl<MAPPING_SHADOW_ADDR>();
}

ALWAYS_INLINE
uptr OriginAddr() {
#if defined(__x86_64__)
  return MappingArchImpl<MAPPING_ORIGIN_ADDR>();
#else
  return 0;
#endif
}

ALWAYS_INLINE
uptr UnionTableAddr() {
  return MappingArchImpl<MAPPING_UNION_TABLE_ADDR>();
}

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
