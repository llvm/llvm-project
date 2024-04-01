// Test hlfir.matmul operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPdot_product1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "lhs"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "rhs"}, %arg2: !fir.ref<i32> {fir.bindc_name = "res"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFdot_product1Elhs"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFdot_product1Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFdot_product1Erhs"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = hlfir.dot_product %0#0 %2#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>) -> i32
  hlfir.assign %3 to %1#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPdot_product1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "lhs"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "rhs"}
// CHECK:           %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "res"}
// CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[ARG2]]

// CHECK-DAG:     %[[LHS_ARG:.*]] = fir.convert %[[LHS_VAR]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[RHS_ARG:.*]] = fir.convert %[[RHS_VAR]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>

// CHECK:         %[[RES_VAL:.*]] = fir.call @_FortranADotProductInteger4(%[[LHS_ARG]], %[[RHS_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]])
// CHECK-NEXT:    hlfir.assign %[[RES_VAL]] to %[[RES_VAR]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPdot_product2(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"}, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFdot_product2Elhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFdot_product2Eres"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFdot_product2Erhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %3 = hlfir.dot_product %0#0 %2#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  hlfir.assign %3 to %1#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @_QPdot_product2(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}
// CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}
// CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[ARG2]]

// CHECK-DAG:     %[[LHS_ARG:.*]] = fir.convert %[[LHS_VAR]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK-DAG:     %[[RHS_ARG:.*]] = fir.convert %[[RHS_VAR]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>

// CHECK:         %[[RES_VAL:.*]] = fir.call @_FortranADotProductLogical(%[[LHS_ARG]], %[[RHS_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]])
// CHECK-NEXT:    %[[CAST:.*]] = fir.convert %[[RES_VAL]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:    hlfir.assign %[[CAST]] to %[[RES_VAR]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

 func.func @_QPdot_product3(%arg0: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "lhs"}, %arg1: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "rhs"}, %arg2: !fir.ref<i32> {fir.bindc_name = "res"}) {
  %c5 = arith.constant 5 : index
  %0 = fir.shape %c5 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFdot_product3Elhs"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
  %2:2 = hlfir.declare %arg2 {uniq_name = "_QFdot_product3Eres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c5_0 = arith.constant 5 : index
  %3 = fir.shape %c5_0 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%3) {uniq_name = "_QFdot_product3Erhs"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
  %5 = hlfir.dot_product %1#0 %4#0 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>) -> i32
  hlfir.assign %5 to %2#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPdot_product3(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "lhs"}
// CHECK:           %[[ARG1:.*]]: !fir.ref<!fir.array<5xi32>> {fir.bindc_name = "rhs"}
// CHECK:           %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "res"}
// CHECK-DAG:     %[[LHS_VAR:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RHS_VAR:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[RES_VAR:.*]]:2 = hlfir.declare %[[ARG2]]

// CHECK-DAG:     %[[LHS_BOX:.*]] = fir.embox %[[LHS_VAR]]#1
// CHECK-DAG:     %[[RHS_BOX:.*]] = fir.embox %[[RHS_VAR]]#1
// CHECK-DAG:     %[[LHS_ARG:.*]] = fir.convert %[[LHS_BOX]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[RHS_ARG:.*]] = fir.convert %[[RHS_BOX]] : (!fir.box<!fir.array<5xi32>>) -> !fir.box<none>

// CHECK:         %[[RES_VAL:.*]] = fir.call @_FortranADotProductInteger4(%[[LHS_ARG]], %[[RHS_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]])
// CHECK-NEXT:    hlfir.assign %[[RES_VAL]] to %[[RES_VAR]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// Test that DOT_PRODUCT's i1 result is casted to !fir.logical.
// Otherwise, the using fir.store becomes invalid.
func.func @_QPdot_product4(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"}, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}) {
  %temp = fir.alloca !fir.logical<4>
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFdot_product2Elhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFdot_product2Erhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %2 = hlfir.dot_product %0#0 %1#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  fir.store %2 to %temp : !fir.ref<!fir.logical<4>>
  return
}
// CHECK-LABEL:   func.func @_QPdot_product4(
// CHECK-SAME:        %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"},
// CHECK-SAME:        %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}) {
// CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.logical<4>
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFdot_product2Elhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QFdot_product2Erhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_3]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           %[[VAL_10:.*]] = fir.convert %[[VAL_4]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.call @_FortranADotProductLogical(%[[VAL_9]], %[[VAL_10]], %{{.*}}, %{{.*}}) fastmath<contract> : (!fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> i1
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i1) -> !fir.logical<4>
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<!fir.logical<4>>
// CHECK:           return
// CHECK:         }
