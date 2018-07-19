//===-- hwasan_mapping.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is a part of HWAddressSanitizer and defines memory mapping.
///
//===----------------------------------------------------------------------===//

#ifndef HWASAN_MAPPING_H
#define HWASAN_MAPPING_H

#include "sanitizer_common/sanitizer_internal_defs.h"

// Typical mapping on Linux/x86_64 with fixed shadow mapping:
// || [0x080000000000, 0x7fffffffffff] || HighMem    ||
// || [0x008000000000, 0x07ffffffffff] || HighShadow ||
// || [0x000100000000, 0x007fffffffff] || ShadowGap  ||
// || [0x000010000000, 0x0000ffffffff] || LowMem     ||
// || [0x000001000000, 0x00000fffffff] || LowShadow  ||
// || [0x000000000000, 0x000000ffffff] || ShadowGap  ||
//
// and with dynamic shadow mapped at [0x770d59f40000, 0x7f0d59f40000]:
// || [0x7f0d59f40000, 0x7fffffffffff] || HighMem    ||
// || [0x7efe2f934000, 0x7f0d59f3ffff] || HighShadow ||
// || [0x7e7e2f934000, 0x7efe2f933fff] || ShadowGap  ||
// || [0x770d59f40000, 0x7e7e2f933fff] || LowShadow  ||
// || [0x000000000000, 0x770d59f3ffff] || LowMem     ||

// Typical mapping on Android/AArch64 (39-bit VMA):
// || [0x001000000000, 0x007fffffffff] || HighMem    ||
// || [0x000800000000, 0x000fffffffff] || ShadowGap  ||
// || [0x000100000000, 0x0007ffffffff] || HighShadow ||
// || [0x000010000000, 0x0000ffffffff] || LowMem     ||
// || [0x000001000000, 0x00000fffffff] || LowShadow  ||
// || [0x000000000000, 0x000000ffffff] || ShadowGap  ||
//
// and with dynamic shadow mapped: [0x007477480000, 0x007c77480000]:
// || [0x007c77480000, 0x007fffffffff] || HighMem    ||
// || [0x007c3ebc8000, 0x007c7747ffff] || HighShadow ||
// || [0x007bbebc8000, 0x007c3ebc7fff] || ShadowGap  ||
// || [0x007477480000, 0x007bbebc7fff] || LowShadow  ||
// || [0x000000000000, 0x00747747ffff] || LowMem     ||

static constexpr __sanitizer::u64 kDefaultShadowSentinel = ~(__sanitizer::u64)0;

// Reasonable values are 4 (for 1/16th shadow) and 6 (for 1/64th).
constexpr __sanitizer::uptr kShadowScale = 4;
constexpr __sanitizer::uptr kShadowAlignment = 1ULL << kShadowScale;

#if SANITIZER_ANDROID
# define HWASAN_FIXED_MAPPING 0
#else
# define HWASAN_FIXED_MAPPING 1
#endif

#if HWASAN_FIXED_MAPPING
# define SHADOW_OFFSET (0)
# define HWASAN_PREMAP_SHADOW 0
#else
# define SHADOW_OFFSET (__hwasan_shadow_memory_dynamic_address)
# define HWASAN_PREMAP_SHADOW 1
#endif

#define SHADOW_GRANULARITY (1ULL << kShadowScale)

#define MEM_TO_SHADOW(mem) (((uptr)(mem) >> kShadowScale) + SHADOW_OFFSET)
#define SHADOW_TO_MEM(shadow) (((uptr)(shadow) - SHADOW_OFFSET) << kShadowScale)

#define MEM_TO_SHADOW_SIZE(size) ((uptr)(size) >> kShadowScale)

#define MEM_IS_APP(mem) MemIsApp((uptr)(mem))

namespace __hwasan {

bool MemIsApp(uptr p);

}  // namespace __hwasan

#endif  // HWASAN_MAPPING_H
