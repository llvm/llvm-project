// RUN: inter-opt %s --inter-resource-info | FileCheck %s
// RUN: inter-opt %s --inter-resource-info -o %t
// RUN: inter-opt %t --inter-resource-info | diff %t -

module {
  func.func @resources() attributes {
      xemachine.barrier_count = 7 : i32,
      xemachine.grf_count = 128 : i32,
      xemachine.grf_used = 1 : i32,
      xemachine.has_global_atomics = false,
      xemachine.has_no_stateless_write = true,
      xemachine.scratch_size = 256 : i64,
      xemachine.slm_size = 128 : i64
    } {
    %root = xemachine.token
    %address = xemachine.archreg 20 : !xemachine.reg<64, 20>
    %data = xemachine.archreg 8 : !xemachine.reg<32, 8>
    %old, %atomic = xemachine.atomic_iadd_a64 %address data %data dep %root
        : (!xemachine.reg<64, 20>, !xemachine.reg<32, 8>)
        -> (!xemachine.reg<32, 14>, !xemachine.mem.token)
    %signal = xemachine.barrier_signal %data dep %atomic
        : !xemachine.reg<32, 8> -> !xemachine.mem.token
    %a = xemachine.archreg 20 : !xemachine.reg<64, 20>
    %b = xemachine.archreg 24 : !xemachine.reg<128, 24>
    %acc = xemachine.archreg 32 : !xemachine.reg<128, 32>
    %result = xemachine.dpas %a, %b, %acc {aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32} : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>, !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
    return
  }

  func.func @no_writes() attributes {
      xemachine.grf_count = 128 : i32,
      xemachine.has_global_atomics = true,
      xemachine.has_no_stateless_write = false
    } {
    %data = xemachine.archreg 8 : !xemachine.reg<32, 8>
    return
  }
}

// CHECK-LABEL: func.func @resources
// CHECK-SAME: xemachine.barrier_count = 1 : i32
// CHECK-SAME: xemachine.grf_count = 128 : i32
// CHECK-SAME: xemachine.grf_used = 40 : i32
// CHECK-SAME: xemachine.has_dpas = true
// CHECK-SAME: xemachine.has_global_atomics = true
// CHECK-SAME: xemachine.has_no_stateless_write = false
// CHECK-SAME: xemachine.scratch_size = 256 : i64
// CHECK-SAME: xemachine.slm_size = 128 : i64
// CHECK-LABEL: func.func @no_writes
// CHECK-SAME: xemachine.barrier_count = 0 : i32
// CHECK-SAME: xemachine.grf_count = 128 : i32
// CHECK-SAME: xemachine.grf_used = 10 : i32
// CHECK-SAME: xemachine.has_dpas = false
// CHECK-SAME: xemachine.has_global_atomics = false
// CHECK-SAME: xemachine.has_no_stateless_write = true
