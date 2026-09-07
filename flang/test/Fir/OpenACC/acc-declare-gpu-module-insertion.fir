// RUN: fir-opt %s -split-input-file --pass-pipeline="builtin.module(acc-declare-gpu-module-insertion{cuda-unified=true},acc-declare-gpu-module-insertion{cuda-unified=true})" | FileCheck %s --check-prefixes=CHECK,UNIFIED
// RUN: fir-opt %s -split-input-file --pass-pipeline="builtin.module(acc-declare-gpu-module-insertion,acc-declare-gpu-module-insertion)" | FileCheck %s --check-prefixes=CHECK,NOUNIF

// Globals registered as host variables must be declarations in device code
// when cuda-unified is set. device_resident globals remain device definitions.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global internal @shared {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<7xf32> {
    %0 = fir.zero_bits !fir.array<7xf32>
    fir.has_value %0 : !fir.array<7xf32>
  }
  fir.global @initialized_shared(dense<1.0> : vector<1xf32>) {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.array<1xf32>
  fir.global @resident {acc.declare = #acc.declare<dataClause = acc_declare_device_resident>} : !fir.array<7xf32> {
    %0 = fir.zero_bits !fir.array<7xf32>
    fir.has_value %0 : !fir.array<7xf32>
  }
  fir.global @precloned {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<1xf32> {
    %0 = fir.zero_bits !fir.array<1xf32>
    fir.has_value %0 : !fir.array<1xf32>
  }
  fir.global @constant_shared {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : i32 {
    %0 = arith.constant 42 : i32
    fir.has_value %0 : i32
  }
  gpu.module @acc_gpu_module {
    fir.global @precloned {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<1xf32> {
      %0 = fir.zero_bits !fir.array<1xf32>
      fir.has_value %0 : !fir.array<1xf32>
    }
  }
}

// UNIFIED-LABEL: gpu.module @acc_gpu_module {
// UNIFIED: fir.global @precloned {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<1xf32>
// UNIFIED-NOT: fir.has_value
// UNIFIED: fir.global @shared {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<7xf32>
// UNIFIED-NOT: fir.has_value
// UNIFIED: fir.global @initialized_shared {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.array<1xf32>
// UNIFIED-NOT: fir.has_value
// UNIFIED: fir.global @resident {{.*}} : !fir.array<7xf32> {
// UNIFIED: fir.has_value
// UNIFIED: fir.global @constant_shared {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : i32 {
// UNIFIED: fir.has_value

// NOUNIF-LABEL: gpu.module @acc_gpu_module {
// NOUNIF: fir.global @precloned {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<1xf32> {
// NOUNIF: fir.has_value
// NOUNIF: fir.global internal @shared {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<7xf32> {
// NOUNIF: fir.has_value
// NOUNIF: fir.global @initialized_shared(dense<1.000000e+00> : vector<1xf32>) {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.array<1xf32>
// NOUNIF: fir.global @resident {{.*}} : !fir.array<7xf32> {
// NOUNIF: fir.has_value
// NOUNIF: fir.global @constant_shared {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : i32 {
// NOUNIF: fir.has_value

// -----

// Device copy of a host linkonce_odr constant keeps that linkage.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global linkonce_odr @param_n {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : i32 {
    %0 = arith.constant 30 : i32
    fir.has_value %0 : i32
  }
}

// CHECK-LABEL: gpu.module @acc_gpu_module {
// CHECK: fir.global linkonce_odr @param_n {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : i32 {
// CHECK: fir.has_value

// -----

// Already present in the GPU module with linkonce; reuse as-is.
module attributes {fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", gpu.container_module} {
  fir.global linkonce @_QQclXlit {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.char<1,3> {
    %0 = fir.string_lit "abc"(3) : !fir.char<1,3>
    fir.has_value %0 : !fir.char<1,3>
  }
  gpu.module @acc_gpu_module {
    fir.global linkonce @_QQclXlit constant : !fir.char<1,3> {
      %0 = fir.string_lit "abc"(3) : !fir.char<1,3>
      fir.has_value %0 : !fir.char<1,3>
    }
  }
}

// CHECK-LABEL: gpu.module @acc_gpu_module {
// CHECK: fir.global linkonce @_QQclXlit {acc.declare = #acc.declare<dataClause = acc_copyin>} constant : !fir.char<1,3> {
// CHECK: fir.has_value
