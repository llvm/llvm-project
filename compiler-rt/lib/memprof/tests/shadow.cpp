//===-- shadow.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "memprof/memprof_mapping.h"

#include "gtest/gtest.h"

// The shadow-mapping macros read this global, which the full runtime normally
// initializes. Provide a definition so the mapping lands in the fake shadow
// below.
extern "C" {
__sanitizer::uptr __memprof_shadow_memory_dynamic_address;
}

namespace __memprof {
namespace {

// A granule-aligned fake "application" address.
static const uptr kFakeMem = 0x40000000;

// Point the shadow so that MEM_TO_SHADOW(kFakeMem) == &Shadow[0].
static void MapShadow(void *Shadow) {
  __memprof_shadow_memory_dynamic_address =
      reinterpret_cast<uptr>(Shadow) - (kFakeMem >> SHADOW_SCALE);
}

TEST(MemprofShadowCount, ExcludesCellPastGranuleAlignedEnd) {
  // [0],[1] cover the 128-byte (two 64B-granule) allocation; [2] is the
  // adjacent block's counter and must NOT be summed.
  alignas(8) u64 Shadow[3] = {10, 20, 999};
  MapShadow(Shadow);
  // 128 is a multiple of MEM_GRANULARITY (64) and kFakeMem is aligned, so
  // p + size lands exactly on the next granule boundary.
  EXPECT_EQ(GetShadowCount(kFakeMem, 128), 30u);
}

TEST(MemprofShadowCount, HistogramExcludesCellPastGranuleAlignedEnd) {
  // [0],[1] cover the 16-byte (two 8B-granule) allocation; [2] is adjacent.
  alignas(8) u8 Shadow[3] = {5, 7, 200};
  MapShadow(Shadow);
  // 16 is a multiple of HISTOGRAM_GRANULARITY (8).
  EXPECT_EQ(GetShadowCountHistogram(kFakeMem, 16), 12u);
}

} // namespace
} // namespace __memprof
