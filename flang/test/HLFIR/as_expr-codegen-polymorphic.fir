// Test hlfir.as_expr codegen for polymorphic expressions.

// RUN: fir-opt %s -bufferize-hlfir | FileCheck %s

!t = !fir.type<t{i:i32}>
func.func @as_expr_class(%arg0 : !fir.class<!t>, %arg1: !fir.ref<!t>) {
    %0 = hlfir.as_expr %arg0 : (!fir.class<!t>) -> !hlfir.expr<!t?>
    hlfir.assign %0 to %arg1 : !hlfir.expr<!t?>, !fir.ref<!t>
   return
}
// CHECK-LABEL:   func.func @as_expr_class(
// CHECK:           %[[VAL_5:.*]] = arith.constant true
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp"} : (!fir.ref<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>>, !fir.ref<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>>)
// ... copy ...
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_6]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>>
// CHECK:           %[[VAL_12:.*]] = fir.undefined tuple<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>, i1>
// CHECK:           %[[VAL_13:.*]] = fir.insert_value %[[VAL_12]], %[[VAL_5]], [1 : index] : (tuple<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>, i1>, i1) -> tuple<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>, i1>
// CHECK:           %[[VAL_14:.*]] = fir.insert_value %[[VAL_13]], %[[VAL_11]], [0 : index] : (tuple<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>, i1>, !fir.class<!fir.heap<!fir.type<t{i:i32}>>>) -> tuple<!fir.class<!fir.heap<!fir.type<t{i:i32}>>>, i1>
// CHECK:           hlfir.assign %[[VAL_11]] to %{{.*}} : !fir.class<!fir.heap<!fir.type<t{i:i32}>>>, !fir.ref<!fir.type<t{i:i32}>>


func.func @as_expr_class_2(%arg0 : !fir.class<!fir.array<?x!t>>) {
    %0 = hlfir.as_expr %arg0 : (!fir.class<!fir.array<?x!t>>) -> !hlfir.expr<?x!t?>
    %c1 = arith.constant 1 : index
    %1 = hlfir.apply %0, %c1 : (!hlfir.expr<?x!t?>, index) -> !hlfir.expr<!t?>
   return
}
// CHECK-LABEL:   func.func @as_expr_class_2(
// CHECK:           %[[VAL_9:.*]] = arith.constant true
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %{{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = ".tmp"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>>) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>>)
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
// ... copy ...
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>>
// CHECK:           %[[VAL_16:.*]] = fir.undefined tuple<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, i1>
// CHECK:           %[[VAL_17:.*]] = fir.insert_value %[[VAL_16]], %[[VAL_9]], [1 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, i1>, i1) -> tuple<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, i1>
// CHECK:           %[[VAL_18:.*]] = fir.insert_value %[[VAL_17]], %[[VAL_15]], [0 : index] : (tuple<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, i1>, !fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>) -> tuple<!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, i1>
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = hlfir.designate %[[VAL_15]] (%[[VAL_19]])  : (!fir.class<!fir.heap<!fir.array<?x!fir.type<t{i:i32}>>>>, index) -> !fir.class<!fir.type<t{i:i32}>>
