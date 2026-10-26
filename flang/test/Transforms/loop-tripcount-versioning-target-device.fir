// RUN: fir-opt --loop-tripcount-versioning %s | FileCheck %s

// Loops that run on a GPU are not versioned (see loop-tripcount-versioning.fir
// for why). When flang compiles the device side of an OpenMP offload program,
// it marks the module with omp.is_target_device and every function in it runs
// on the GPU, whether or not the function has its own omp.declare_target
// attribute. This file checks that the module attribute alone is enough to
// stop versioning. It is separate from the main test because the attribute
// has to be on an explicit module.
module attributes {omp.is_target_device = true} {
  func.func @declared(%n: index) attributes {omp.declare_target = #omp.declaretarget<device_type = any, capture_clause = enter>} {
    %a = fir.alloca !fir.array<4xi32>
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    fir.do_loop %i = %c1 to %n step %c1 {
      %first = arith.cmpi eq, %i, %c1 : index
      fir.if %first { nvvm.barrier }
      %p = fir.array_coor %a %i : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
      fir.store %c0_i32 to %p : !fir.ref<i32>
    }
    return
  }
  // CHECK-LABEL: func.func @declared(
  // CHECK-NOT:     arith.constant 4
  // CHECK:         nvvm.barrier
  // CHECK-NOT:     nvvm.barrier
  // CHECK:         return

  // Reached from device code without a declare target attribute of its own.
  func.func @implicit(%a: !fir.ref<!fir.array<4xi32>>, %n: index) {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    fir.do_loop %i = %c1 to %n step %c1 {
      %p = fir.array_coor %a %i : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
      fir.store %c0_i32 to %p : !fir.ref<i32>
    }
    return
  }
  // CHECK-LABEL: func.func @implicit(
  // CHECK-NOT:     fir.if
  // CHECK:         return
}
