// Test hlfir.as_expr codegen for polymorphic expressions.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

!t = !fir.type<t{i:i32}>
func.func @as_expr_class(%arg0 : !fir.class<!t>, %arg1: !fir.ref<!t>) {
    %0 = hlfir.as_expr %arg0 : (!fir.class<!t>) -> !hlfir.expr<!t?>
    hlfir.assign %0 to %arg1 : !hlfir.expr<!t?>, !fir.ref<!t>
   return
}
// CHECK-LABEL:   func.func @as_expr_class(
// CHECK:           %[[VAL_6:.*]]:2 =  hlfir.declare %{{.*}} {uniq_name = ".tmp"} : (!fir.class<!fir.heap<!fir.type<t{i:i32}>>>) -> (!fir.class<!fir.type<t{i:i32}>>, !fir.class<!fir.type<t{i:i32}>>)
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// ... copy ...
// CHECK:           %[[VAL_12:.*]] = fir.undefined tuple<!fir.class<!fir.type<t{i:i32}>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_5]], [1 : index] : (tuple<!fir.class<!fir.type<t{i:i32}>>, i1>, i1) -> tuple<!fir.class<!fir.type<t{i:i32}>>, i1>
// CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_6]]#0, [0 : index] : (tuple<!fir.class<!fir.type<t{i:i32}>>, i1>, !fir.class<!fir.type<t{i:i32}>>) -> tuple<!fir.class<!fir.type<t{i:i32}>>, i1>
// CHECK:           hlfir.assign %[[VAL_6]]#0 to %{{.*}} : !fir.class<!fir.type<t{i:i32}>>, !fir.ref<!fir.type<t{i:i32}>>


func.func @as_expr_class_2(%arg0 : !fir.class<!fir.array<?x!t>>) {
    %0 = hlfir.as_expr %arg0 : (!fir.class<!fir.array<?x!t>>) -> !hlfir.expr<?x!t?>
    %c1 = arith.constant 1 : index
    %1 = hlfir.apply %0, %c1 : (!hlfir.expr<?x!t?>, index) -> !hlfir.expr<!t?>
   return
}
// CHECK-LABEL:   func.func @as_expr_class_2(
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = ".tmp"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>) -> (!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, !fir.class<!fir.array<?x!fir.type<t{i:i32}>>>)
// CHECK:           %[[VAL_9:.*]] = arith.constant true
// ... copy ...
// CHECK:           %[[VAL_16:.*]] = fir.undefined tuple<!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, i1>
// CHECK:           %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_9]], [1 : index] : (tuple<!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, i1>, i1) -> tuple<!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, i1>
// CHECK:           %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_10]]#0, [0 : index] : (tuple<!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, i1>, !fir.class<!fir.array<?x!fir.type<t{i:i32}>>>) -> tuple<!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, i1>
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_19]])  : (!fir.class<!fir.array<?x!fir.type<t{i:i32}>>>, index) -> !fir.class<!fir.type<t{i:i32}>>
