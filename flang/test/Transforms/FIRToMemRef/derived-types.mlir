// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @load_scalar
// CHECK:       [[INDEX:%[0-9]]] = fir.field_index
// CHECK:       [[COORD:%[0-9]]] = fir.coordinate_of
// CHECK-NEXT:  [[CONVERT:%[0-9]]] = fir.convert [[COORD]] : (!fir.ref<f32>) -> memref<f32>
// CHECK-NEXT:  memref.load [[CONVERT]][] : memref<f32>
func.func @load_scalar(%arg0: !fir.ref<!fir.type<DerivedType{a:f32}>>) {
    %0 = fir.undefined !fir.dscope
    %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "dtype"} : (!fir.ref<!fir.type<DerivedType{a:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<DerivedType{a:f32}>>
    %2 = fir.field_index a, !fir.type<DerivedType{a:f32}>
    %3 = fir.coordinate_of %1, %2 : (!fir.ref<!fir.type<DerivedType{a:f32}>>, !fir.field) -> !fir.ref<f32>
    %4 = fir.load %3 : !fir.ref<f32>
    return
}

// CHECK-LABEL: func.func @store_scalar
// CHECK:       [[INDEX:%[0-9]]] = fir.field_index
// CHECK:       [[COORD:%[0-9]]] = fir.coordinate_of
// CHECK-NEXT:  [[CONVERT:%[0-9]]] = fir.convert [[COORD]] : (!fir.ref<f32>) -> memref<f32>
// CHECK-NEXT:  memref.store %4, [[CONVERT]][] : memref<f32>
func.func @store_scalar(%arg0: !fir.ref<!fir.type<DerivedType{a:f32}>>, %arg1: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.declare %arg0 dummy_scope %0 {uniq_name = "dtype"} : (!fir.ref<!fir.type<DerivedType{a:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<DerivedType{a:f32}>>
  %2 = fir.declare %arg1 dummy_scope %0 {uniq_name = "x"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %3 = fir.load %2 : !fir.ref<f32>
  %4 = fir.field_index a, !fir.type<DerivedType{a:f32}>
  %5 = fir.coordinate_of %1, %4 : (!fir.ref<!fir.type<DerivedType{a:f32}>>, !fir.field) -> !fir.ref<f32>
  fir.store %3 to %5 : !fir.ref<f32>
  return
}
