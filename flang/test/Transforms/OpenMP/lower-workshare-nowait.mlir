// RUN: fir-opt --split-input-file --lower-workshare --allow-unregistered-dialect %s | FileCheck %s

// Check that we correctly handle nowait

// CHECK-LABEL:   func.func @nonowait
func.func @nonowait(%arg0: !fir.ref<!fir.array<42xi32>>) {
  // CHECK: omp.barrier
  omp.workshare {
    omp.terminator
  }
  return
}

// -----

// CHECK-LABEL:   func.func @nowait
func.func @nowait(%arg0: !fir.ref<!fir.array<42xi32>>) {
  // CHECK-NOT: omp.barrier
  omp.workshare nowait {
    omp.terminator
  }
  return
}
