// RUN: fir-opt --split-input-file --omp-indirect-call-lookup %s | FileCheck %s

// On a GPU target device, indirect calls are routed through the lookup so the
// host callee address is resolved to the device address.

// CHECK: func.func private @__llvm_omp_indirect_call_lookup(() -> ()) -> (() -> ())
module attributes {omp.is_target_device = true, omp.is_gpu = true} {
  // CHECK-LABEL: func.func @gpu_device_caller
  func.func @gpu_device_caller(%arg0: !fir.ref<!fir.boxproc<() -> i32>>) -> i32 {
    %0 = fir.load %arg0 : !fir.ref<!fir.boxproc<() -> i32>>
    %1 = fir.box_addr %0 : (!fir.boxproc<() -> i32>) -> (() -> i32)
    // CHECK: %[[BOXADDR:.*]] = fir.box_addr
    // CHECK: %[[HOST:.*]] = fir.convert %[[BOXADDR]] : (() -> i32) -> (() -> ())
    // CHECK: %[[DEV:.*]] = fir.call @__llvm_omp_indirect_call_lookup(%[[HOST]]) : (() -> ()) -> (() -> ())
    // CHECK: %[[CALLEE:.*]] = fir.convert %[[DEV]] : (() -> ()) -> (() -> i32)
    // CHECK: fir.call %[[CALLEE]]() : () -> i32
    %2 = fir.call %1() : () -> i32
    return %2 : i32
  }
}

// -----

// A non-GPU target device shares the host address space, so no lookup is added.
module attributes {omp.is_target_device = true, omp.is_gpu = false} {
  // CHECK-LABEL: func.func @nongpu_device_caller
  // CHECK-NOT: __llvm_omp_indirect_call_lookup
  func.func @nongpu_device_caller(%arg0: !fir.ref<!fir.boxproc<() -> i32>>) -> i32 {
    %0 = fir.load %arg0 : !fir.ref<!fir.boxproc<() -> i32>>
    %1 = fir.box_addr %0 : (!fir.boxproc<() -> i32>) -> (() -> i32)
    // CHECK: fir.call %{{.*}}() : () -> i32
    %2 = fir.call %1() : () -> i32
    return %2 : i32
  }
}

// -----

// Host compilation: no lookup is added.
module attributes {omp.is_target_device = false, omp.is_gpu = false} {
  // CHECK-LABEL: func.func @host_caller
  // CHECK-NOT: __llvm_omp_indirect_call_lookup
  func.func @host_caller(%arg0: !fir.ref<!fir.boxproc<() -> i32>>) -> i32 {
    %0 = fir.load %arg0 : !fir.ref<!fir.boxproc<() -> i32>>
    %1 = fir.box_addr %0 : (!fir.boxproc<() -> i32>) -> (() -> i32)
    %2 = fir.call %1() : () -> i32
    return %2 : i32
  }
}

// -----

// Direct calls are left untouched on a GPU target device.
module attributes {omp.is_target_device = true, omp.is_gpu = true} {
  func.func private @callee() -> i32
  // CHECK-LABEL: func.func @direct_caller
  // CHECK-NOT: __llvm_omp_indirect_call_lookup
  func.func @direct_caller() -> i32 {
    // CHECK: fir.call @callee() : () -> i32
    %0 = fir.call @callee() : () -> i32
    return %0 : i32
  }
}
