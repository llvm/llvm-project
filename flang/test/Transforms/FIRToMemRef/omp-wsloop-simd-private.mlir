// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// Verifies that fir-to-memref does not insert any operations in-between
// LoopWrapperInterface ops which violates the single operation invariant enforced
// by the interface.

omp.private {type = private} @_QFEa_private_i32 : i32

// CHECK-LABEL: func.func @_QQmain
// CHECK: [[DECL:%.*]] = fir.declare
// CHECK: [[CONV1:%.*]] = fir.convert [[DECL]]
// CHECK: omp.parallel
// CHECK-NEXT: {{%.*}} = fir.convert [[CONV1]]
// CHECK-NEXT: omp.wsloop

func.func @_QQmain() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFEa"}
  %1 = fir.declare %0 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c0_i32 to %1 : !fir.ref<i32>
  omp.parallel {
    omp.wsloop {
      omp.simd private(@_QFEa_private_i32 %1 -> %arg0 : !fir.ref<i32>) {
        omp.loop_nest (%arg1) : i32 = (%c1_i32) to (%c1_i32) inclusive step (%c1_i32) {
          omp.yield
        }
      } {omp.composite}
    } {omp.composite}
    omp.terminator
  }
  return
}
