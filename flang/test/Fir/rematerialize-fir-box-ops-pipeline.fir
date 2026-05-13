// RUN: tco -test-gen %s | FileCheck %s

// Test the real FIR-to-LLVM pipeline. In particular, this guards against
// adding a FIR-level CSE-like pass between fir-rematerialize-box-ops and
// FIR-to-LLVM lowering, which could merge FIR box operations across OpenMP
// regions before their descriptor allocas are materialized.

func.func private @use_box(!fir.box<!fir.array<?xi32>>)

func.func @box_used_in_sibling_tasks(%arg0: !fir.ref<!fir.array<?xi32>>, %n: index) {
  %shape = fir.shape %n : (index) -> !fir.shape<1>
  %box = fir.embox %arg0(%shape) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
  omp.task {
    fir.call @use_box(%box) : (!fir.box<!fir.array<?xi32>>) -> ()
    omp.terminator
  }
  omp.task {
    fir.call @use_box(%box) : (!fir.box<!fir.array<?xi32>>) -> ()
    omp.terminator
  }
  return
}

// CHECK-LABEL: llvm.func @box_used_in_sibling_tasks(
// CHECK:       omp.task {
// CHECK:         llvm.alloca
// CHECK:         llvm.call @use_box
// CHECK:         omp.terminator
// CHECK:       omp.task {
// CHECK:         llvm.alloca
// CHECK:         llvm.call @use_box
// CHECK:         omp.terminator
