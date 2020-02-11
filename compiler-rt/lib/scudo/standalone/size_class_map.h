//===-- size_class_map.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_SIZE_CLASS_MAP_H_
#define SCUDO_SIZE_CLASS_MAP_H_

#include "chunk.h"
#include "common.h"
#include "string_utils.h"

namespace scudo {

inline uptr scaledLog2(uptr Size, uptr ZeroLog, uptr LogBits) {
  const uptr L = getMostSignificantSetBitIndex(Size);
  const uptr LBits = (Size >> (L - LogBits)) - (1 << LogBits);
  const uptr HBits = (L - ZeroLog) << LogBits;
  return LBits + HBits;
}

template <typename Config> struct SizeClassMapBase {
  static u32 getMaxCachedHint(uptr Size) {
    DCHECK_LE(Size, (1UL << Config::MaxSizeLog) + Chunk::getHeaderSize());
    DCHECK_NE(Size, 0);
    u32 N;
    // Force a 32-bit division if the template parameters allow for it.
    if (Config::MaxBytesCachedLog > 31 || Config::MaxSizeLog > 31)
      N = static_cast<u32>((1UL << Config::MaxBytesCachedLog) / Size);
    else
      N = (1U << Config::MaxBytesCachedLog) / static_cast<u32>(Size);
    return Max(1U, Min(Config::MaxNumCachedHint, N));
  }
};

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
template <typename Config>
class FixedSizeClassMap : public SizeClassMapBase<Config> {
  typedef SizeClassMapBase<Config> Base;

  static const uptr MinSize = 1UL << Config::MinSizeLog;
  static const uptr MidSize = 1UL << Config::MidSizeLog;
  static const uptr MidClass = MidSize / MinSize;
  static const u8 S = Config::NumBits - 1;
  static const uptr M = (1UL << S) - 1;

  static const uptr SizeDelta = Chunk::getHeaderSize();

public:
  static const u32 MaxNumCachedHint = Config::MaxNumCachedHint;

  static const uptr MaxSize = (1UL << Config::MaxSizeLog) + SizeDelta;
  static const uptr NumClasses =
      MidClass + ((Config::MaxSizeLog - Config::MidSizeLog) << S) + 1;
  static_assert(NumClasses <= 256, "");
  static const uptr LargestClassId = NumClasses - 1;
  static const uptr BatchClassId = 0;

  static uptr getSizeByClassId(uptr ClassId) {
    DCHECK_NE(ClassId, BatchClassId);
    if (ClassId <= MidClass)
      return (ClassId << Config::MinSizeLog) + SizeDelta;
    ClassId -= MidClass;
    const uptr T = MidSize << (ClassId >> S);
    return T + (T >> S) * (ClassId & M) + SizeDelta;
  }

  static uptr getClassIdBySize(uptr Size) {
    if (Size <= SizeDelta + (1 << Config::MinSizeLog))
      return 1;
    Size -= SizeDelta;
    DCHECK_LE(Size, MaxSize);
    if (Size <= MidSize)
      return (Size + MinSize - 1) >> Config::MinSizeLog;
    return MidClass + 1 + scaledLog2(Size - 1, Config::MidSizeLog, S);
  }
};

template <typename Config>
class TableSizeClassMap : public SizeClassMapBase<Config> {
  static const u8 S = Config::NumBits - 1;
  static const uptr M = (1UL << S) - 1;
  static const uptr ClassesSize =
      sizeof(Config::Classes) / sizeof(Config::Classes[0]);

  struct SizeTable {
    constexpr SizeTable() {
      uptr Pos = 1 << Config::MidSizeLog;
      uptr Inc = 1 << (Config::MidSizeLog - S);
      for (uptr i = 0; i != getTableSize(); ++i) {
        Pos += Inc;
        if ((Pos & (Pos - 1)) == 0)
          Inc *= 2;
        Tab[i] = computeClassId(Pos + Config::SizeDelta);
      }
    }

    constexpr static u8 computeClassId(uptr Size) {
      for (uptr i = 0; i != ClassesSize; ++i) {
        if (Size <= Config::Classes[i])
          return static_cast<u8>(i + 1);
      }
      return static_cast<u8>(-1);
    }

    constexpr static uptr getTableSize() {
      return (Config::MaxSizeLog - Config::MidSizeLog) << S;
    }

    u8 Tab[getTableSize()] = {};
  };

  static constexpr SizeTable Table = {};

public:
  static const u32 MaxNumCachedHint = Config::MaxNumCachedHint;

  static const uptr NumClasses = ClassesSize + 1;
  static_assert(NumClasses < 256, "");
  static const uptr LargestClassId = NumClasses - 1;
  static const uptr BatchClassId = 0;
  static const uptr MaxSize = Config::Classes[LargestClassId - 1];

  static uptr getSizeByClassId(uptr ClassId) {
    return Config::Classes[ClassId - 1];
  }

  static uptr getClassIdBySize(uptr Size) {
    if (Size <= Config::Classes[0])
      return 1;
    Size -= Config::SizeDelta;
    DCHECK_LE(Size, MaxSize);
    if (Size <= (1 << Config::MidSizeLog))
      return ((Size - 1) >> Config::MinSizeLog) + 1;
    return Table.Tab[scaledLog2(Size - 1, Config::MidSizeLog, S)];
  }

  static void print() {}
  static void validate() {}
};

struct AndroidSizeClassConfig {
#if SCUDO_WORDSIZE == 64U
  // Measured using a system_server profile.
  static const uptr NumBits = 7;
  static const uptr MinSizeLog = 4;
  static const uptr MidSizeLog = 6;
  static const uptr MaxSizeLog = 16;
  static const u32 MaxNumCachedHint = 14;
  static const uptr MaxBytesCachedLog = 14;

  static constexpr u32 Classes[] = {
      0x00020, 0x00030, 0x00040, 0x00050, 0x00060, 0x00070, 0x00090, 0x000a0,
      0x000b0, 0x000e0, 0x00110, 0x00130, 0x001a0, 0x00240, 0x00320, 0x00430,
      0x00640, 0x00830, 0x00a10, 0x00c30, 0x01010, 0x01150, 0x01ad0, 0x02190,
      0x03610, 0x04010, 0x04510, 0x04d10, 0x05a10, 0x07310, 0x09610, 0x10010,
  };
  static const uptr SizeDelta = 16;
#else
  // Measured using a dex2oat profile.
  static const uptr NumBits = 8;
  static const uptr MinSizeLog = 4;
  static const uptr MidSizeLog = 8;
  static const uptr MaxSizeLog = 16;
  static const u32 MaxNumCachedHint = 14;
  static const uptr MaxBytesCachedLog = 14;

  static constexpr u32 Classes[] = {
      0x00020, 0x00030, 0x00040, 0x00050, 0x00060, 0x00070, 0x00080, 0x00090,
      0x000a0, 0x000b0, 0x000c0, 0x000d0, 0x000e0, 0x000f0, 0x00100, 0x00110,
      0x00120, 0x00140, 0x00150, 0x00170, 0x00190, 0x001c0, 0x001f0, 0x00220,
      0x00240, 0x00260, 0x002a0, 0x002e0, 0x00310, 0x00340, 0x00380, 0x003b0,
      0x003e0, 0x00430, 0x00490, 0x00500, 0x00570, 0x005f0, 0x00680, 0x00720,
      0x007d0, 0x00890, 0x00970, 0x00a50, 0x00b80, 0x00cb0, 0x00e30, 0x00fb0,
      0x011b0, 0x01310, 0x01470, 0x01790, 0x01b50, 0x01fd0, 0x02310, 0x02690,
      0x02b10, 0x02fd0, 0x03610, 0x03e10, 0x04890, 0x05710, 0x06a90, 0x10010,
  };
  static const uptr SizeDelta = 16;
#endif
};

typedef TableSizeClassMap<AndroidSizeClassConfig> AndroidSizeClassMap;

struct DefaultSizeClassConfig {
  static const uptr NumBits = 3;
  static const uptr MinSizeLog = 5;
  static const uptr MidSizeLog = 8;
  static const uptr MaxSizeLog = 17;
  static const u32 MaxNumCachedHint = 8;
  static const uptr MaxBytesCachedLog = 10;
};

typedef FixedSizeClassMap<DefaultSizeClassConfig> DefaultSizeClassMap;

struct SvelteSizeClassConfig {
#if SCUDO_WORDSIZE == 64U
  static const uptr NumBits = 4;
  static const uptr MinSizeLog = 4;
  static const uptr MidSizeLog = 8;
  static const uptr MaxSizeLog = 14;
  static const u32 MaxNumCachedHint = 4;
  static const uptr MaxBytesCachedLog = 10;
#else
  static const uptr NumBits = 4;
  static const uptr MinSizeLog = 3;
  static const uptr MidSizeLog = 7;
  static const uptr MaxSizeLog = 14;
  static const u32 MaxNumCachedHint = 5;
  static const uptr MaxBytesCachedLog = 10;
#endif
};

typedef FixedSizeClassMap<SvelteSizeClassConfig> SvelteSizeClassMap;

template <typename SCMap> inline void printMap() {
  ScopedString Buffer(1024);
  uptr PrevS = 0;
  uptr TotalCached = 0;
  for (uptr I = 0; I < SCMap::NumClasses; I++) {
    if (I == SCMap::BatchClassId)
      continue;
    const uptr S = SCMap::getSizeByClassId(I);
    const uptr D = S - PrevS;
    const uptr P = PrevS ? (D * 100 / PrevS) : 0;
    const uptr L = S ? getMostSignificantSetBitIndex(S) : 0;
    const uptr Cached = SCMap::getMaxCachedHint(S) * S;
    Buffer.append(
        "C%02zu => S: %zu diff: +%zu %02zu%% L %zu Cached: %zu %zu; id %zu\n",
        I, S, D, P, L, SCMap::getMaxCachedHint(S), Cached,
        SCMap::getClassIdBySize(S));
    TotalCached += Cached;
    PrevS = S;
  }
  Buffer.append("Total Cached: %zu\n", TotalCached);
  Buffer.output();
}

template <typename SCMap> static void validateMap() {
  for (uptr C = 0; C < SCMap::NumClasses; C++) {
    if (C == SCMap::BatchClassId)
      continue;
    const uptr S = SCMap::getSizeByClassId(C);
    CHECK_NE(S, 0U);
    CHECK_EQ(SCMap::getClassIdBySize(S), C);
    if (C < SCMap::LargestClassId)
      CHECK_EQ(SCMap::getClassIdBySize(S + 1), C + 1);
    CHECK_EQ(SCMap::getClassIdBySize(S - 1), C);
    if (C - 1 != SCMap::BatchClassId)
      CHECK_GT(SCMap::getSizeByClassId(C), SCMap::getSizeByClassId(C - 1));
  }
  // Do not perform the loop if the maximum size is too large.
  if (SCMap::MaxSize > (1 << 19))
    return;
  for (uptr S = 1; S <= SCMap::MaxSize; S++) {
    const uptr C = SCMap::getClassIdBySize(S);
    CHECK_LT(C, SCMap::NumClasses);
    CHECK_GE(SCMap::getSizeByClassId(C), S);
    if (C - 1 != SCMap::BatchClassId)
      CHECK_LT(SCMap::getSizeByClassId(C - 1), S);
  }
}
} // namespace scudo

#endif // SCUDO_SIZE_CLASS_MAP_H_
