// RUN: %clangxx_lowfat -O3 %s -o %t && %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISS
// RUN: %clangxx_lowfat_safe -O3 %s -o %t && not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-CATCH

// Demonstrates a behavioral difference between Fast mode and Safe/Comprehensive mode.
//
// The scenario: a noinline function performs an OOB heap read but its return value
// is discarded by the caller.
//
// Fast mode has no PipelineStartEP pass. LLVM's Dead Argument Elimination (DAE)
// sees peek()'s return value is unused at all call sites and rewrites:
//   ret %loaded_val  →  ret undef
// The load becomes dead and is DCE'd. The LowFat pass at OptimizerLastEP finds
// no load to instrument — the OOB is missed.
//
// Safe mode's barrier pass runs at PipelineStartEP and inserts:
//   1. @llvm.sideeffect() — prevents call-level DCE by blocking memory(none)
//      inference on peek().
//   2. @llvm.fake.use(loaded_val) — after every load. fake.use creates a data
//      dependency on the loaded value without emitting machine code,
//      preventing DAE from marking the return value as dead. The load survives
//      to OptimizerLastEP where LowFat instruments it → OOB is caught.
//
// Comprehensive mode instruments at PipelineStartEP before any optimization,
// so the check call is inserted before DAE has a chance to remove the load.
//
// Expected outputs:
//   Fast (-O3):          load DCE'd by DAE → no OOB check → program exits 0 → DONE
//   Safe (-O3):          fake.use keeps load alive → OOB detected → LOWFAT ERROR

#include <cstdio>
#include <cstdlib>

// noinline: the optimizer cannot see the body from the call site in Fast mode,
// so it performs inter-procedural attribute inference rather than inlining.
__attribute__((noinline))
static double peek(char *p) {
  // 8-byte (double) OOB read. p was allocated with malloc(16); a double starting
  // at offset 14 spans bytes [14, 22), which overflows the 16-byte LowFat slot
  // boundary at byte 16. LowFat detects this as an out-of-bounds access.
  return *reinterpret_cast<double *>(p + 14);
}

int main() {
  char *p = (char *)malloc(16);
  peek(p);   // Return value discarded — caller has no use for it.
  free(p);

  // CHECK-MISS: DONE
  // CHECK-CATCH: LOWFAT ERROR: out-of-bounds error detected!
  printf("DONE\n");
  return 0;
}
