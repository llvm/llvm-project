// Test alias analysis for POINTER + FIRSTPRIVATE in omp.private copy region.
// For POINTER variables, the descriptor is copied but pointer association
// is preserved. Both the mold descriptor and private descriptor point to
// the SAME target data, therefore they ALIAS.
//
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-alias-analysis)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// CHECK-LABEL: Testing : <<NULL ATTRIBUTE>>
// CHECK: mold_data#0 <-> priv_data#0: MayAlias

// This test verifies that alias analysis correctly returns MayAlias for POINTER
// variables in omp.private copy regions. For POINTER, the descriptor is copied
// but pointer association is preserved, so both descriptors point to the SAME
// target data.

omp.private {type = firstprivate} @ptr_privatizer : !fir.box<!fir.ptr<!fir.array<8xi32>>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>):
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %0 = fir.zero_bits !fir.ptr<!fir.array<8xi32>>
  %1 = fir.shape %c8 : (index) -> !fir.shape<1>
  %2 = fir.embox %0(%1) : (!fir.ptr<!fir.array<8xi32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<8xi32>>>
  fir.store %2 to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>)
} copy {
^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>, %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>):
  // For POINTER: the descriptor is copied, but pointer association is preserved.
  // Load both the mold (%arg0) and private (%arg1) to test alias analysis.
  // Both descriptors point to THE SAME target data => should ALIAS.
  %mold_desc = fir.load %arg0 {test.ptr = "mold_data"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>
  fir.store %mold_desc to %arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>
  %priv_desc = fir.load %arg1 {test.ptr = "priv_data"} : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>
  omp.yield(%arg1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<8xi32>>>>)
}
