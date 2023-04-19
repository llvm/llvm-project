// RUN: tco %s | FileCheck %s

func.func @array_coor_box_value(%29 : !fir.box<!fir.array<2xf64>>,
     			   %33 : index) -> f64 {
  %34 = fir.array_coor %29 %33 : (!fir.box<!fir.array<2xf64>>, index) ->
      		       	          !fir.ref<f64>
  %35 = fir.load %34 : !fir.ref<f64>
  return %35 : f64
}

// CHECK-LABEL: define double @array_coor_box_value
// CHECK: %[[t3:.*]] = sub i64 %{{.*}}, 1
// CHECK: %[[t4:.*]] = mul i64 %[[t3]], 1
// CHECK: %[[t5:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 7, i32 0, i32 2
// CHECK: %[[t6:.*]] = load i64, ptr %[[t5]]
// CHECK: %[[t7:.*]] = mul i64 %[[t4]], %[[t6]]
// CHECK: %[[t8:.*]] = add i64 %[[t7]], 0
// CHECK: %[[t9:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %{{.*}}, i32 0, i32 0
// CHECK: %[[t10:.*]] = load ptr, ptr %[[t9]]
// CHECK: %[[t11:.*]] = getelementptr i8, ptr %[[t10]], i64 %[[t8]]
// CHECK: %[[t12:.*]] = load double, ptr %[[t11]]
// CHECK: ret double %[[t12]]

func.func @test_array_coor_box_component_slice(%arg0: !fir.box<!fir.array<2x!fir.type<t{i:i32,j:i32}>>>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %0 = fir.field_index j, !fir.type<t{i:i32,j:i32}>
  %1 = fir.slice %c1, %c2, %c1 path %0 : (index, index, index, !fir.field) -> !fir.slice<1>
  %2 = fir.array_coor %arg0 [%1] %c2 : (!fir.box<!fir.array<2x!fir.type<t{i:i32,j:i32}>>>, !fir.slice<1>, index) -> !fir.ref<i32>
  fir.call @take_int(%2) : (!fir.ref<i32>) -> ()
  return
}
func.func private @take_int(%arg0: !fir.ref<i32>) -> ()

// CHECK-LABEL: define void @test_array_coor_box_component_slice(
// CHECK-SAME: ptr %[[VAL_0:.*]])
// CHECK:   %[[VAL_1:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[VAL_0]], i32 0, i32 7, i32 0, i32 2
// CHECK:   %[[VAL_2:.*]] = load i64, ptr %[[VAL_1]]
// CHECK:   %[[VAL_3:.*]] = mul i64 1, %[[VAL_2]]
// CHECK:   %[[VAL_4:.*]] = add i64 %[[VAL_3]], 0
// CHECK:   %[[VAL_5:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]], ptr, [1 x i64] }, ptr %[[VAL_0]], i32 0, i32 0
// CHECK:   %[[VAL_6:.*]] = load ptr, ptr %[[VAL_5]]
// CHECK:   %[[VAL_7:.*]] = getelementptr i8, ptr %[[VAL_6]], i64 %[[VAL_4]]
// CHECK:   %[[VAL_8:.*]] = getelementptr %t, ptr %[[VAL_7]], i32 0, i32 1
// CHECK:   call void @take_int(ptr %[[VAL_8]])
