// RUN: fir-opt --cse -split-input-file %s | FileCheck %s

// Check that the redundant fir.load is removed.
func.func @fun(%arg0: !fir.ref<i64>) -> i64 {
    %0 = fir.load %arg0 : !fir.ref<i64>
    %1 = fir.load %arg0 : !fir.ref<i64>
    %2 = arith.addi %0, %1 : i64
    return %2 : i64
}

// CHECK-LABEL: func @fun
// CHECK-NEXT:    %[[LOAD:.*]] = fir.load %{{.*}} : !fir.ref<i64>
// CHECK-NEXT:    %{{.*}} = arith.addi %[[LOAD]], %[[LOAD]] : i64

// -----

// CHECK-LABEL: func @fun(
// CHECK-SAME:            %[[A:.*]]: !fir.ref<i64>
func.func @fun(%a : !fir.ref<i64>) -> i64 {
  // CHECK: %[[LOAD:.*]] = fir.load %[[A]] : !fir.ref<i64>
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi %[[LOAD]], %[[LOAD]] : i64
  %3 = arith.addi %1, %2 : i64
  %4 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %5 = arith.addi %3, %4 : i64
  %6 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %7 = arith.addi %5, %6 : i64
  %8 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %9 = arith.addi %7, %8 : i64
  %10 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %11 = arith.addi %10, %9 : i64
  %12 = fir.load %a : !fir.ref<i64>
  // CHECK-NEXT: %{{.*}} = arith.addi
  %13 = arith.addi %11, %12 : i64
  // CHECK-NEXT: return %{{.*}} : i64
  return %13 : i64
}

// -----

func.func @fun(%a : !fir.ref<i64>) -> i64 {
  cf.br ^bb1
^bb1:
  %1 = fir.load %a : !fir.ref<i64>
  %2 = fir.load %a : !fir.ref<i64>
  %3 = arith.addi %1, %2 : i64
  cf.br ^bb2
^bb2:
  %4 = fir.load %a : !fir.ref<i64>
  %5 = arith.subi %4, %4 : i64
  return %5 : i64
}

// -----

// Check that the redundant ops on volatile operands are PRESERVED.
func.func @fun(%arg0: !fir.ref<i64, volatile>) -> i64 {
    %0 = fir.load %arg0 : !fir.ref<i64, volatile>
    %1 = fir.load %arg0 : !fir.ref<i64, volatile>
    %2 = arith.addi %0, %1 : i64
    fir.store %2 to %arg0 : !fir.ref<i64, volatile>
    fir.store %2 to %arg0 : !fir.ref<i64, volatile>
    return %2 : i64
}
// CHECK-LABEL:   func.func @fun(%arg0: !fir.ref<i64, volatile>) -> i64 {
// CHECK:           %[[VAL_1:.*]] = fir.load %arg0 : !fir.ref<i64, volatile>
// CHECK:           %[[VAL_2:.*]] = fir.load %arg0 : !fir.ref<i64, volatile>
// CHECK:           %[[VAL_3:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           fir.store %[[VAL_3]] to %arg0 : !fir.ref<i64, volatile>
// CHECK:           fir.store %[[VAL_3]] to %arg0 : !fir.ref<i64, volatile>
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         }

// -----

// Check that volatile hlfir assignments are PRESERVED.
func.func @_QPdot_product2(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"}, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.volatile_cast %arg0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x!fir.logical<4>>, volatile>
  %2:2 = hlfir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Elhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>)
  %3 = fir.volatile_cast %arg2 : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
  %4:2 = hlfir.declare %3 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Eres"} : (!fir.ref<!fir.logical<4>, volatile>, !fir.dscope) -> (!fir.ref<!fir.logical<4>, volatile>, !fir.ref<!fir.logical<4>, volatile>)
  %5 = fir.volatile_cast %arg1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x!fir.logical<4>>, volatile>
  %6:2 = hlfir.declare %5 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Erhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>)
  %7 = hlfir.dot_product %2#0 %6#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>) -> !fir.logical<4>
  hlfir.assign %7 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  %8 = hlfir.dot_product %2#0 %6#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>) -> !fir.logical<4>
  hlfir.assign %8 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
  return
}

// CHECK-LABEL:   func.func @_QPdot_product2(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "lhs"},
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "rhs"},
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "res"}) {
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x!fir.logical<4>>, volatile>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Elhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>)
// CHECK:           %[[VAL_6:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>, volatile>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_6]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Eres"} : (!fir.ref<!fir.logical<4>, volatile>, !fir.dscope) -> (!fir.ref<!fir.logical<4>, volatile>, !fir.ref<!fir.logical<4>, volatile>)
// CHECK:           %[[VAL_8:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<!fir.array<?x!fir.logical<4>>, volatile>
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_8]] dummy_scope %[[VAL_3]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFdot_product2Erhs"} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>)
// CHECK:           %[[VAL_10:.*]] = hlfir.dot_product %[[VAL_5]]#0 %[[VAL_9]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>) -> !fir.logical<4>
// CHECK:           hlfir.assign %[[VAL_10]] to %[[VAL_7]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
// CHECK:           %[[VAL_11:.*]] = hlfir.dot_product %[[VAL_5]]#0 %[[VAL_9]]#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>, volatile>, !fir.box<!fir.array<?x!fir.logical<4>>, volatile>) -> !fir.logical<4>
// CHECK:           hlfir.assign %[[VAL_11]] to %[[VAL_7]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>, volatile>
// CHECK:           return
// CHECK:         }
