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

void initializeWorkingSet() {
  // The shadow mapping assumes 64 so this cannot be changed.
  CHECK(getFlags()->cache_line_size == 64);
}

int finalizeWorkingSet() {
  // FIXME NYI: we need to add memory scanning to report the total lines
  // touched, and later add sampling to get intermediate values.
  Report("%s is not finished: nothing yet to report\n", SanitizerToolName);
  return 0;
}

} // namespace __esan
