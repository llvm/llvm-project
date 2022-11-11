// Test fir.field_index llvm code generation
// RUN: fir-opt %s | tco | FileCheck %s


// CHECK-DAG: %[[a:.*]] = type { float, i32 }
// CHECK-DAG: %[[b:.*]] = type { float, i32 }
// CHECK-DAG: %[[c:.*]] = type { float, %[[b]] }

// CHECK-LABEL: @simple_field
// CHECK-SAME: (ptr %[[arg0:.*]])
func.func @simple_field(%arg0: !fir.ref<!fir.type<a{x:f32,i:i32}>>) -> i32 {
  %1 = fir.field_index i, !fir.type<a{x:f32,i:i32}>
  // CHECK: %[[GEP:.*]] = getelementptr %a, ptr %[[arg0]], i32 0, i32 1
  %2 = fir.coordinate_of %arg0, %1 : (!fir.ref<!fir.type<a{x:f32,i:i32}>>, !fir.field) -> !fir.ref<i32>
  // CHECK: load i32, ptr %[[GEP]]
  %3 = fir.load %2 : !fir.ref<i32>
  return %3 : i32
}

// CHECK-LABEL: @derived_field
// CHECK-SAME: (ptr %[[arg0:.*]])
func.func @derived_field(%arg0: !fir.ref<!fir.type<c{x:f32,some_b:!fir.type<b{x:f32,i:i32}>}>>) -> i32 {
  %1 = fir.field_index some_b, !fir.type<c{x:f32,some_b:!fir.type<b{x:f32,i:i32}>}>
  %2 = fir.field_index i, !fir.type<b{x:f32,i:i32}>
  // CHECK: %[[GEP:.*]] = getelementptr %c, ptr %[[arg0]], i32 0, i32 1, i32 1
  %3 = fir.coordinate_of %arg0, %1, %2 : (!fir.ref<!fir.type<c{x:f32,some_b:!fir.type<b{x:f32,i:i32}>}>>, !fir.field, !fir.field) -> !fir.ref<i32>
  // CHECK: load i32, ptr %[[GEP]]
  %4 = fir.load %3 : !fir.ref<i32>
  return %4 : i32
}
