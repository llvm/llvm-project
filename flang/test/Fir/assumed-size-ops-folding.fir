// RUN: fir-opt --canonicalize %s | FileCheck %s

// Verify: fir.is_assumed_size_extent(fir.assumed_size_extent) folds to i1 true.

// CHECK-LABEL: func.func @fold(
func.func @fold() -> i1 {
  %e = fir.assumed_size_extent : index
  // CHECK: %[[C:.*]] = arith.constant true
  %t = fir.is_assumed_size_extent %e : (index) -> i1
  return %t : i1
}


