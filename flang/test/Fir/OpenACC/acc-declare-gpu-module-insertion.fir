// RUN: fir-opt %s --pass-pipeline="builtin.module(acc-declare-gpu-module-insertion{cuda-unified=true},acc-declare-gpu-module-insertion{cuda-unified=true})" | FileCheck %s

// Globals registered as host variables must be declarations in device code.
// device_resident globals remain device definitions.
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
}

// CHECK-LABEL: gpu.module @acc_gpu_module {
// CHECK: fir.global @shared {acc.declare = #acc.declare<dataClause = acc_create>} : !fir.array<7xf32>
// CHECK-NOT: fir.has_value
// CHECK: fir.global @initialized_shared {acc.declare = #acc.declare<dataClause = acc_copyin>} : !fir.array<1xf32>
// CHECK-NOT: fir.has_value
// CHECK: fir.global @resident {{.*}} : !fir.array<7xf32> {
// CHECK: fir.has_value
