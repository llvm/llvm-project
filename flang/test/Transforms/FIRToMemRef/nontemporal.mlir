// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// The `nontemporal` flag set on fir.load/fir.store by the lower-nontemporal
// pass has to reach the memref ops, otherwise the `omp simd nontemporal`
// hint is lost before codegen.

// CHECK-LABEL: func.func @nontemporal_scalar
// CHECK:       memref.load %{{.*}}[] nontemporal(true) : memref<f32>
// CHECK:       memref.store %{{.*}}, %{{.*}}[] nontemporal(true) : memref<f32>
func.func @nontemporal_scalar(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 uniq_name("a") : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %2 = fir.declare %arg1 dummy_scope %0 uniq_name("b") : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %3 = fir.load %1 {nontemporal} : !fir.ref<f32>
  fir.store %3 to %2 {nontemporal} : !fir.ref<f32>
  return
}

// Accesses that are not marked keep the default.

// CHECK-LABEL: func.func @plain_scalar
// CHECK:       memref.load %{{.*}}[] : memref<f32>
// CHECK-NOT:   nontemporal
// CHECK:       memref.store %{{.*}}, %{{.*}}[] : memref<f32>
func.func @plain_scalar(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 uniq_name("a") : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %2 = fir.declare %arg1 dummy_scope %0 uniq_name("b") : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %3 = fir.load %1 : !fir.ref<f32>
  fir.store %3 to %2 : !fir.ref<f32>
  return
}
