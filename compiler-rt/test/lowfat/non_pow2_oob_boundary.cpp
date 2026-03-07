// RUN: %clangxx_lowfat -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_lowfat_safe -O1 %s -o %t && not %run %t 2>&1 | FileCheck %s

// Demonstrates GEP-level (pointer arithmetic) instrumentation catching an OOB
// pointer that escapes to a neighbouring allocation slot — a case where
// load/store checking alone produces a false negative.
//
// SCENARIO
// --------
// p is a 48-byte LowFat allocation at address k*48 (absolute-zero aligned).
// p+48 = (k+1)*48 is the exact start of the NEXT 48-byte slot.
//
//   Load/store check on *(p+48):
//     GetBase(p+48) = (k+1)*48 = p+48  <-- attributed to NEXT slot
//     End           = p+48 + 48 = p+96
//     AccessEnd     = p+48 + 1  = p+49
//     p+49 <= p+96  --> NOT OOB  (false negative — sink() would not catch it)
//
//   GEP check at the arithmetic (p + 48) itself:
//     GetBase(p)    = k*48 = p           <-- uses SOURCE pointer's bounds
//     End           = p + 48
//     result p+48   >= End               --> OOB (detected before escape!)
//
// By checking at the point of pointer arithmetic, the error is caught before
// the OOB pointer reaches sink(), regardless of which slot it happens to land in.
//
// REQUIRES: lowfat-custom-config

#include <stdlib.h>

// Prevent inlining so the OOB pointer is forced to cross a function boundary,
// making the "escape" explicit. The store inside sink() is NOT detectable by
// load/store checking alone (GetBase(q) = q, so it appears in-bounds).
__attribute__((noinline))
static void sink(volatile char *q) { *q = 'x'; }

int main() {
  char *p = (char *)malloc(48);
  if (!p) return 1;

  // GEP check: p+48 is computed using p's bounds (End = p+48).
  // result == End  -->  OOB detected here, before the pointer reaches sink().
  // CHECK: LOWFAT ERROR: out-of-bounds error detected!
  sink(p + 48);

  free(p);
  return 0;
}

