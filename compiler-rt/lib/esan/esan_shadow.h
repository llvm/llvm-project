//===-- esan_shadow.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Shadow memory mappings for the esan run-time.
//===----------------------------------------------------------------------===//

#ifndef ESAN_SHADOW_H
#define ESAN_SHADOW_H

#include <sanitizer_common/sanitizer_platform.h>

#if SANITIZER_WORDSIZE != 64
#error Only 64-bit is supported
#endif

namespace __esan {

#if SANITIZER_LINUX && defined(__x86_64__)
// Linux x86_64
//
// Application memory falls into these 5 regions (ignoring the corner case
// of PIE with a non-zero PT_LOAD base):
//
// [0x00000000'00000000, 0x00000100'00000000) non-PIE + heap
// [0x00005500'00000000, 0x00005700'00000000) PIE
// [0x00007e00'00000000, 0x00007fff'ff600000) libraries + stack, part 1
// [0x00007fff'ff601000, 0x00008000'00000000) libraries + stack, part 2
// [0xffffffff'ff600000, 0xffffffff'ff601000) vsyscall
//
// Although we can ignore the vsyscall for the most part as there are few data
// references there (other sanitizers ignore it), we enforce a gap inside the
// library region to distinguish the vsyscall's shadow, considering this gap to
// be an invalid app region.
// We disallow application memory outside of those 5 regions.
// Our regions assume that the stack rlimit is less than a terabyte (otherwise
// the Linux kernel's default mmap region drops below 0x7e00'), which we enforce
// at init time (we can support larger and unlimited sizes for shadow
// scaledowns, but it is difficult for 1:1 mappings).
//
// Our shadow memory is scaled from a 1:1 mapping and supports a scale
// specified at library initialization time that can be any power-of-2
// scaledown (1x, 2x, 4x, 8x, 16x, etc.).
//
// We model our shadow memory after Umbra, a library used by the Dr. Memory
// tool: https://github.com/DynamoRIO/drmemory/blob/master/umbra/umbra_x64.c.
// We use Umbra's scheme as it was designed to support different
// offsets, it supports two different shadow mappings (which we may want to
// use for future tools), and it ensures that the shadow of a shadow will
// not overlap either shadow memory or application memory.
//
// This formula translates from application memory to shadow memory:
//
//   shadow(app) = ((app & 0x00000fff'ffffffff) + offset) >> scale
//
// Where the offset for 1:1 is 0x00001300'00000000.  For other scales, the
// offset is shifted left by the scale, except for scales of 1 and 2 where
// it must be tweaked in order to pass the double-shadow test
// (see the "shadow(shadow)" comments below):
//   scale == 0: 0x00001300'000000000
//   scale == 1: 0x00002200'000000000
//   scale == 2: 0x00004400'000000000
//   scale >= 3: (0x00001300'000000000 << scale)
//
// Do not pass in the open-ended end value to the formula as it will fail.
//
// The resulting shadow memory regions for a 0 scaling are:
//
// [0x00001300'00000000, 0x00001400'00000000)
// [0x00001800'00000000, 0x00001a00'00000000)
// [0x00002100'00000000, 0x000022ff'ff600000)
// [0x000022ff'ff601000, 0x00002300'00000000)
// [0x000022ff'ff600000, 0x000022ff'ff601000]
//
// We also want to ensure that a wild access by the application into the shadow
// regions will not corrupt our own shadow memory.  shadow(shadow) ends up
// disjoint from shadow(app):
//
// [0x00001600'00000000, 0x00001700'00000000)
// [0x00001b00'00000000, 0x00001d00'00000000)
// [0x00001400'00000000, 0x000015ff'ff600000]
// [0x000015ff'ff601000, 0x00001600'00000000]
// [0x000015ff'ff600000, 0x000015ff'ff601000]

struct ApplicationRegion {
  uptr Start;
  uptr End;
  bool ShadowMergedWithPrev;
};

static const struct ApplicationRegion AppRegions[] = {
  {0x0000000000000000ull, 0x0000010000000000u, false},
  {0x0000550000000000u,   0x0000570000000000u, false},
  // We make one shadow mapping to hold the shadow regions for all 3 of these
  // app regions, as the mappings interleave, and the gap between the 3rd and
  // 4th scales down below a page.
  {0x00007e0000000000u,   0x00007fffff600000u, false},
  {0x00007fffff601000u,   0x0000800000000000u, true},
  {0xffffffffff600000u,   0xffffffffff601000u, true},
};
static const u32 NumAppRegions = sizeof(AppRegions)/sizeof(AppRegions[0]);

// See the comment above: we do not currently support a stack size rlimit
// equal to or larger than 1TB.
static const uptr MaxStackSize = (1ULL << 40) - 4096;

class ShadowMapping {
public:
  static const uptr Mask = 0x00000fffffffffffu;
  // The scale and offset vary by tool.
  uptr Scale;
  uptr Offset;
  void initialize(uptr ShadowScale) {
    static const uptr OffsetArray[3] = {
        0x0000130000000000u,
        0x0000220000000000u,
        0x0000440000000000u,
    };
    Scale = ShadowScale;
    if (Scale <= 2)
      Offset = OffsetArray[Scale];
    else
      Offset = OffsetArray[0] << Scale;
  }
};
extern ShadowMapping Mapping;
#else
// We'll want to use templatized functions over the ShadowMapping once
// we support more platforms.
#error Platform not supported
#endif

static inline bool getAppRegion(u32 i, uptr *Start, uptr *End) {
  if (i >= NumAppRegions)
    return false;
  *Start = AppRegions[i].Start;
  *End = AppRegions[i].End;
  return true;
}

ALWAYS_INLINE
bool isAppMem(uptr Mem) {
  for (u32 i = 0; i < NumAppRegions; ++i) {
    if (Mem >= AppRegions[i].Start && Mem < AppRegions[i].End)
      return true;
  }
  return false;
}

ALWAYS_INLINE
uptr appToShadow(uptr App) {
  return (((App & ShadowMapping::Mask) + Mapping.Offset) >> Mapping.Scale);
}

static inline bool getShadowRegion(u32 i, uptr *Start, uptr *End) {
  if (i >= NumAppRegions)
    return false;
  u32 UnmergedShadowCount = 0;
  u32 AppIdx;
  for (AppIdx = 0; AppIdx < NumAppRegions; ++AppIdx) {
    if (!AppRegions[AppIdx].ShadowMergedWithPrev) {
      if (UnmergedShadowCount == i)
        break;
      UnmergedShadowCount++;
    }
  }
  if (AppIdx >= NumAppRegions || UnmergedShadowCount != i)
    return false;
  *Start = appToShadow(AppRegions[AppIdx].Start);
  // The formula fails for the end itself.
  *End = appToShadow(AppRegions[AppIdx].End - 1) + 1;
  // Merge with adjacent shadow regions:
  for (++AppIdx; AppIdx < NumAppRegions; ++AppIdx) {
    if (!AppRegions[AppIdx].ShadowMergedWithPrev)
      break;
    *Start = Min(*Start, appToShadow(AppRegions[AppIdx].Start));
    *End = Max(*End, appToShadow(AppRegions[AppIdx].End - 1) + 1);
  }
  return true;
}

ALWAYS_INLINE
bool isShadowMem(uptr Mem) {
  // We assume this is not used on any critical performance path and so there's
  // no need to hardcode the mapping results.
  for (uptr i = 0; i < NumAppRegions; ++i) {
    if (Mem >= appToShadow(AppRegions[i].Start) &&
        Mem < appToShadow(AppRegions[i].End - 1) + 1)
      return true;
  }
  return false;
}

} // namespace __esan

#endif /* ESAN_SHADOW_H */
