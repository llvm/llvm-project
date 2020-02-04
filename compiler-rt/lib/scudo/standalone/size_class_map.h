//===-- size_class_map.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_SIZE_CLASS_MAP_H_
#define SCUDO_SIZE_CLASS_MAP_H_

#include "common.h"
#include "string_utils.h"

namespace scudo {

// SizeClassMap maps allocation sizes into size classes and back, in an
// efficient table-free manner.
//
// Class 0 is a special class that doesn't abide by the same rules as other
// classes. The allocator uses it to hold batches.
//
// The other sizes are controlled by the template parameters:
// - MinSizeLog: defines the first class as 2^MinSizeLog bytes.
// - MaxSizeLog: defines the last class as 2^MaxSizeLog bytes.
// - MidSizeLog: classes increase with step 2^MinSizeLog from 2^MinSizeLog to
//               2^MidSizeLog bytes.
// - NumBits: the number of non-zero bits in sizes after 2^MidSizeLog.
//            eg. with NumBits==3 all size classes after 2^MidSizeLog look like
//            0b1xx0..0 (where x is either 0 or 1).
//
// This class also gives a hint to a thread-caching allocator about the amount
// of chunks that can be cached per-thread:
// - MaxNumCachedHint is a hint for the max number of chunks cached per class.
// - 2^MaxBytesCachedLog is the max number of bytes cached per class.

template <u8 NumBits, u8 MinSizeLog, u8 MidSizeLog, u8 MaxSizeLog,
          u32 MaxNumCachedHintT, u8 MaxBytesCachedLog>
class SizeClassMap {
  static const uptr MinSize = 1UL << MinSizeLog;
  static const uptr MidSize = 1UL << MidSizeLog;
  static const uptr MidClass = MidSize / MinSize;
  static const u8 S = NumBits - 1;
  static const uptr M = (1UL << S) - 1;

public:
  static const u32 MaxNumCachedHint = MaxNumCachedHintT;

  static const uptr MaxSize = 1UL << MaxSizeLog;
  static const uptr NumClasses =
      MidClass + ((MaxSizeLog - MidSizeLog) << S) + 1;
  static_assert(NumClasses <= 256, "");
  static const uptr LargestClassId = NumClasses - 1;
  static const uptr BatchClassId = 0;

  static uptr getSizeByClassId(uptr ClassId) {
    DCHECK_NE(ClassId, BatchClassId);
    if (ClassId <= MidClass)
      return ClassId << MinSizeLog;
    ClassId -= MidClass;
    const uptr T = MidSize << (ClassId >> S);
    return T + (T >> S) * (ClassId & M);
  }

  static uptr getClassIdBySize(uptr Size) {
    DCHECK_LE(Size, MaxSize);
    if (Size <= MidSize)
      return (Size + MinSize - 1) >> MinSizeLog;
    Size -= 1;
    const uptr L = getMostSignificantSetBitIndex(Size);
    const uptr LBits = (Size >> (L - S)) - (1 << S);
    const uptr HBits = (L - MidSizeLog) << S;
    return MidClass + 1 + HBits + LBits;
  }

  static u32 getMaxCachedHint(uptr Size) {
    DCHECK_LE(Size, MaxSize);
    DCHECK_NE(Size, 0);
    u32 N;
    // Force a 32-bit division if the template parameters allow for it.
    if (MaxBytesCachedLog > 31 || MaxSizeLog > 31)
      N = static_cast<u32>((1UL << MaxBytesCachedLog) / Size);
    else
      N = (1U << MaxBytesCachedLog) / static_cast<u32>(Size);
    return Max(1U, Min(MaxNumCachedHint, N));
  }

  static void print() {
    ScopedString Buffer(1024);
    uptr PrevS = 0;
    uptr TotalCached = 0;
    for (uptr I = 0; I < NumClasses; I++) {
      if (I == BatchClassId)
        continue;
      const uptr S = getSizeByClassId(I);
      if (S >= MidSize / 2 && (S & (S - 1)) == 0)
        Buffer.append("\n");
      const uptr D = S - PrevS;
      const uptr P = PrevS ? (D * 100 / PrevS) : 0;
      const uptr L = S ? getMostSignificantSetBitIndex(S) : 0;
      const uptr Cached = getMaxCachedHint(S) * S;
      Buffer.append(
          "C%02zu => S: %zu diff: +%zu %02zu%% L %zu Cached: %zu %zu; id %zu\n",
          I, getSizeByClassId(I), D, P, L, getMaxCachedHint(S), Cached,
          getClassIdBySize(S));
      TotalCached += Cached;
      PrevS = S;
    }
    Buffer.append("Total Cached: %zu\n", TotalCached);
    Buffer.output();
  }

  static void validate() {
    for (uptr C = 0; C < NumClasses; C++) {
      if (C == BatchClassId)
        continue;
      const uptr S = getSizeByClassId(C);
      CHECK_NE(S, 0U);
      CHECK_EQ(getClassIdBySize(S), C);
      if (C < LargestClassId)
        CHECK_EQ(getClassIdBySize(S + 1), C + 1);
      CHECK_EQ(getClassIdBySize(S - 1), C);
      if (C - 1 != BatchClassId)
        CHECK_GT(getSizeByClassId(C), getSizeByClassId(C - 1));
    }
    // Do not perform the loop if the maximum size is too large.
    if (MaxSizeLog > 19)
      return;
    for (uptr S = 1; S <= MaxSize; S++) {
      const uptr C = getClassIdBySize(S);
      CHECK_LT(C, NumClasses);
      CHECK_GE(getSizeByClassId(C), S);
      if (C - 1 != BatchClassId)
        CHECK_LT(getSizeByClassId(C - 1), S);
    }
  }
};

typedef SizeClassMap<3, 5, 8, 17, 8, 10> DefaultSizeClassMap;

// TODO(kostyak): further tune class maps for Android & Fuchsia.
#if SCUDO_WORDSIZE == 64U
typedef SizeClassMap<4, 4, 8, 14, 4, 10> SvelteSizeClassMap;
typedef SizeClassMap<2, 5, 9, 16, 14, 14> AndroidSizeClassMap;
#else
typedef SizeClassMap<4, 3, 7, 14, 5, 10> SvelteSizeClassMap;
typedef SizeClassMap<2, 5, 9, 16, 14, 14> AndroidSizeClassMap;
#endif

} // namespace scudo

#endif // SCUDO_SIZE_CLASS_MAP_H_
