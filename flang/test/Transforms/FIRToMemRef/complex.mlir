// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @convert_complex_
// CHECK:       [[DECL0:%[0-9]+]] = fir.declare %arg0
// CHECK:       [[DECL1:%[0-9]+]] = fir.declare %arg1
// CHECK:       [[CONVERT1:%[0-9]+]] = fir.convert [[DECL1]] : (!fir.ref<complex<f32>>) -> memref<complex<f32>>
// CHECK:       [[LOAD:%[0-9]+]] = memref.load [[CONVERT1]][] : memref<complex<f32>>
// CHECK:       [[CONVERT0:%[0-9]+]] = fir.convert [[DECL0]] : (!fir.ref<complex<f32>>) -> memref<complex<f32>>
// CHECK:       memref.store [[LOAD]], [[CONVERT0]][] : memref<complex<f32>>

func.func @convert_complex_(%arg0: !fir.ref<complex<f32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<complex<f32>> {fir.bindc_name = "b"}) attributes {fir.internal_name = "_QPconvert_complex"} {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFconvert_complexEa"} : (!fir.ref<complex<f32>>, !fir.dscope) -> !fir.ref<complex<f32>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "_QFconvert_complexEb"} : (!fir.ref<complex<f32>>, !fir.dscope) -> !fir.ref<complex<f32>>
  %3 = fir.load %2 : !fir.ref<complex<f32>>
  fir.store %3 to %1 : !fir.ref<complex<f32>>
  return
}
