// Use --mlir-disable-threading so that the diagnostic printing is serialized.
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

module {
  // Build a scalar view via fir.declare with a storage operand into an array of i8
  func.func @_QPdeclare_with_storage_is_nonscalar() {
    %c0 = arith.constant 0 : index
    %arr = fir.alloca !fir.array<4xi8>
    %elem_i8 = fir.coordinate_of %arr, %c0 : (!fir.ref<!fir.array<4xi8>>, index) -> !fir.ref<i8>
    %elem_f32 = fir.convert %elem_i8 : (!fir.ref<i8>) -> !fir.ref<f32>
    %view = fir.declare %elem_f32 storage(%arr[0]) {uniq_name = "_QFpi"}
      : (!fir.ref<f32>, !fir.ref<!fir.array<4xi8>>) -> !fir.ref<f32>
    // Force interface query through an acc op that prints type category
    %cp = acc.copyin varPtr(%view : !fir.ref<f32>) -> !fir.ref<f32> {name = "pi", structured = false}
    acc.enter_data dataOperands(%cp : !fir.ref<f32>)
    return
  }

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<f32>) -> !fir.ref<f32> {name = "pi", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<f32>
  // CHECK: Type category: array
}


