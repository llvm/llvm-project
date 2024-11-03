// Test hlfir.parent_comp code generation to FIR
// RUN: fir-opt %s -convert-hlfir-to-fir | FileCheck %s

func.func @test_scalar(%arg0: !fir.ref<!fir.type<t2{i:i32,j:i32}>>) {
  %1 = hlfir.parent_comp %arg0 : (!fir.ref<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<t1{i:i32}>>
  return
}
// CHECK-LABEL:   func.func @test_scalar(
// CHECK-SAME:   %[[VAL_0:.*]]: !fir.ref<!fir.type<t2{i:i32,j:i32}>>) {
// CHECK:  fir.convert %[[VAL_0]] : (!fir.ref<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<t1{i:i32}>>

func.func @test_scalar_polymorphic(%arg0: !fir.class<!fir.type<t2{i:i32,j:i32}>>) {
  %1 = hlfir.parent_comp %arg0 : (!fir.class<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<t1{i:i32}>>
  return
}
// CHECK-LABEL:   func.func @test_scalar_polymorphic(
// CHECK-SAME:               %[[VAL_0:.*]]: !fir.class<!fir.type<t2{i:i32,j:i32}>>) {
// CHECK:  %[[VAL_1:.*]] = fir.box_addr %[[VAL_0]] : (!fir.class<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<t2{i:i32,j:i32}>>
// CHECK:  fir.convert %[[VAL_1]] : (!fir.ref<!fir.type<t2{i:i32,j:i32}>>) -> !fir.ref<!fir.type<t1{i:i32}>>

func.func @test_array(%arg0: !fir.ref<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = hlfir.parent_comp %arg0 shape %1 : (!fir.ref<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<t1{i:i32}>>>
  return
}
// CHECK-LABEL:   func.func @test_array(
// CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  %[[VAL_3:.*]] = fir.embox %[[VAL_0]](%[[VAL_2]]) : (!fir.ref<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>
// CHECK:  fir.rebox %[[VAL_3]] : (!fir.box<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) -> !fir.box<!fir.array<10x!fir.type<t1{i:i32}>>>

func.func @test_array_polymorphic(%arg0: !fir.class<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = hlfir.parent_comp %arg0 shape %1 : (!fir.class<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>, !fir.shape<1>) -> !fir.box<!fir.array<10x!fir.type<t1{i:i32}>>>
  return
}
// CHECK-LABEL:   func.func @test_array_polymorphic(
// CHECK-SAME:              %[[VAL_0:.*]]: !fir.class<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) {
// CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:  %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:  fir.rebox %[[VAL_0]] : (!fir.class<!fir.array<10x!fir.type<t2{i:i32,j:i32}>>>) -> !fir.box<!fir.array<10x!fir.type<t1{i:i32}>>>
