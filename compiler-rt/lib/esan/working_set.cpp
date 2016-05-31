//===-- working_set.cpp ---------------------------------------------------===//
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
// This file contains working-set-specific code.
//===----------------------------------------------------------------------===//

#include "working_set.h"
#include "esan.h"
#include "esan_flags.h"
#include "esan_shadow.h"
#include "sanitizer_common/sanitizer_procmaps.h"

// We shadow every cache line of app memory with one shadow byte.
// - The highest bit of each shadow byte indicates whether the corresponding
//   cache line has ever been accessed.
// - The lowest bit of each shadow byte indicates whether the corresponding
//   cache line was accessed since the last sample.
// - The other bits can be used either for a single working set snapshot
//   between two consecutive samples, or an aggregate working set snapshot
//   over multiple sample periods (future work).
// We live with races in accessing each shadow byte.
typedef unsigned char byte;

namespace __esan {

// Our shadow memory assumes that the line size is 64.
static const u32 CacheLineSize = 64;

// See the shadow byte layout description above.
static const u32 TotalWorkingSetBitIdx = 7;
static const u32 CurWorkingSetBitIdx = 0;
static const byte ShadowAccessedVal =
  (1 << TotalWorkingSetBitIdx) | (1 << CurWorkingSetBitIdx);

void processRangeAccessWorkingSet(uptr PC, uptr Addr, SIZE_T Size,
                                  bool IsWrite) {
  if (Size == 0)
    return;
  SIZE_T I = 0;
  uptr LineSize = getFlags()->cache_line_size;
  // As Addr+Size could overflow at the top of a 32-bit address space,
  // we avoid the simpler formula that rounds the start and end.
  SIZE_T NumLines = Size / LineSize +
    // Add any extra at the start or end adding on an extra line:
    (LineSize - 1 + Addr % LineSize + Size % LineSize) / LineSize;
  byte *Shadow = (byte *)appToShadow(Addr);
  // Write shadow bytes until we're word-aligned.
  while (I < NumLines && (uptr)Shadow % 4 != 0) {
    if ((*Shadow & ShadowAccessedVal) != ShadowAccessedVal)
      *Shadow |= ShadowAccessedVal;
    ++Shadow;
    ++I;
  }
  // Write whole shadow words at a time.
  // Using a word-stride loop improves the runtime of a microbenchmark of
  // memset calls by 10%.
  u32 WordValue = ShadowAccessedVal | ShadowAccessedVal << 8 |
    ShadowAccessedVal << 16 | ShadowAccessedVal << 24;
  while (I + 4 <= NumLines) {
    if ((*(u32*)Shadow & WordValue) != WordValue)
      *(u32*)Shadow |= WordValue;
    Shadow += 4;
    I += 4;
  }
  // Write any trailing shadow bytes.
  while (I < NumLines) {
    if ((*Shadow & ShadowAccessedVal) != ShadowAccessedVal)
      *Shadow |= ShadowAccessedVal;
    ++Shadow;
    ++I;
  }
}

// This routine will word-align ShadowStart and ShadowEnd prior to scanning.
static u32 countAndClearShadowValues(u32 BitIdx, uptr ShadowStart,
                                     uptr ShadowEnd) {
  u32 WorkingSetSize = 0;
  u32 ByteValue = 0x1 << BitIdx;
  u32 WordValue = ByteValue | ByteValue << 8 | ByteValue << 16 |
    ByteValue << 24;
  // Get word aligned start.
  ShadowStart = RoundDownTo(ShadowStart, sizeof(u32));
  for (u32 *Ptr = (u32 *)ShadowStart; Ptr < (u32 *)ShadowEnd; ++Ptr) {
    if ((*Ptr & WordValue) != 0) {
      byte *BytePtr = (byte *)Ptr;
      for (u32 j = 0; j < sizeof(u32); ++j) {
        if (BytePtr[j] & ByteValue) {
          ++WorkingSetSize;
          // TODO: Accumulate to the lower-frequency bit to the left.
        }
      }
      // Clear this bit from every shadow byte.
      *Ptr &= ~WordValue;
    }
  }
  return WorkingSetSize;
}

// Scan shadow memory to calculate the number of cache lines being accessed,
// i.e., the number of non-zero bits indexed by BitIdx in each shadow byte.
// We also clear the lowest bits (most recent working set snapshot).
static u32 computeWorkingSizeAndReset(u32 BitIdx) {
  u32 WorkingSetSize = 0;
  MemoryMappingLayout MemIter(true/*cache*/);
  uptr Start, End, Prot;
  while (MemIter.Next(&Start, &End, nullptr/*offs*/, nullptr/*file*/,
                      0/*file size*/, &Prot)) {
    VPrintf(4, "%s: considering %p-%p app=%d shadow=%d prot=%u\n",
            __FUNCTION__, Start, End, Prot, isAppMem(Start),
            isShadowMem(Start));
    if (isShadowMem(Start) && (Prot & MemoryMappingLayout::kProtectionWrite)) {
      VPrintf(3, "%s: walking %p-%p\n", __FUNCTION__, Start, End);
      WorkingSetSize += countAndClearShadowValues(BitIdx, Start, End);
    }
  }
  return WorkingSetSize;
}

void initializeWorkingSet() {
  CHECK(getFlags()->cache_line_size == CacheLineSize);
  registerMemoryFaultHandler();
}

static u32 getSizeForPrinting(u32 NumOfCachelines, const char *&Unit) {
  // We need a constant to avoid software divide support:
  static const u32 KilobyteCachelines = (0x1 << 10) / CacheLineSize;
  static const u32 MegabyteCachelines = KilobyteCachelines << 10;

  if (NumOfCachelines > 10 * MegabyteCachelines) {
    Unit = "MB";
    return NumOfCachelines / MegabyteCachelines;
  } else if (NumOfCachelines > 10 * KilobyteCachelines) {
    Unit = "KB";
    return NumOfCachelines / KilobyteCachelines;
  } else {
    Unit = "Bytes";
    return NumOfCachelines * CacheLineSize;
  }
}

int finalizeWorkingSet() {
  // Get the working set size for the entire execution.
  u32 NumOfCachelines = computeWorkingSizeAndReset(TotalWorkingSetBitIdx);
  const char *Unit;
  u32 Size = getSizeForPrinting(NumOfCachelines, Unit);
  Report(" %s: the total working set size: %u %s (%u cache lines)\n",
         SanitizerToolName, Size, Unit, NumOfCachelines);
  return 0;
}

} // namespace __esan
