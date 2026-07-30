// RUN: fir-opt %s --fir-select-ops-conversion | FileCheck %s

// Exercises all five case-attribute kinds for fir.select_case with an i32
// selector: #fir.upper, #fir.point, #fir.interval, #fir.lower, and unit.

func.func @select_case_integer(%arg0: !fir.ref<i32>) -> i32 {
  %2 = fir.load %arg0 : !fir.ref<i32>
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c4_i32 = arith.constant 4 : i32
  %c5_i32 = arith.constant 5 : i32
  %c7_i32 = arith.constant 7 : i32
  %c8_i32 = arith.constant 8 : i32
  %c15_i32 = arith.constant 15 : i32
  %c21_i32 = arith.constant 21 : i32
  fir.select_case %2 : i32 [#fir.upper, %c1_i32, ^bb1,
                            #fir.point, %c2_i32, ^bb2,
                            #fir.interval, %c4_i32, %c5_i32, ^bb4,
                            #fir.point, %c7_i32, ^bb5,
                            #fir.interval, %c8_i32, %c15_i32, ^bb5,
                            #fir.lower, %c21_i32, ^bb5,
                            unit, ^bb3]
^bb1:  // pred: ^bb0
  %c1_i32_0 = arith.constant 1 : i32
  fir.store %c1_i32_0 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb2:  // pred: ^bb0
  %c2_i32_1 = arith.constant 2 : i32
  fir.store %c2_i32_1 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb3:  // pred: ^bb0
  %c0_i32 = arith.constant 0 : i32
  fir.store %c0_i32 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb4:  // pred: ^bb0
  %c4_i32_2 = arith.constant 4 : i32
  fir.store %c4_i32_2 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb5:  // 3 preds: ^bb0, ^bb0, ^bb0
  %c7_i32_3 = arith.constant 7 : i32
  fir.store %c7_i32_3 to %arg0 : !fir.ref<i32>
  cf.br ^bb6
^bb6:  // 5 preds: ^bb1, ^bb2, ^bb3, ^bb4, ^bb5
  %3 = fir.load %arg0 : !fir.ref<i32>
  return %3 : i32
}

// CHECK-LABEL: func.func @select_case_integer(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.ref<i32>) -> i32 {
// CHECK:         %[[SELECT_VALUE:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
// CHECK:         %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:         %[[CST2:.*]] = arith.constant 2 : i32
// CHECK:         %[[CST4:.*]] = arith.constant 4 : i32
// CHECK:         %[[CST5:.*]] = arith.constant 5 : i32
// CHECK:         %[[CST7:.*]] = arith.constant 7 : i32
// CHECK:         %[[CST8:.*]] = arith.constant 8 : i32
// CHECK:         %[[CST15:.*]] = arith.constant 15 : i32
// CHECK:         %[[CST21:.*]] = arith.constant 21 : i32
// Check for upper bound `case (:1)`
// CHECK:         %[[CMP_SLE:.*]] = arith.cmpi sle, %[[SELECT_VALUE]], %[[CST1]] : i32
// CHECK:         cf.cond_br %[[CMP_SLE]], ^[[BB_UPPER:bb[0-9]+]], ^bb1
// CHECK-LABEL: ^bb1:
// Check for point value `case (2)`
// CHECK:         %[[CMP_EQ:.*]] = arith.cmpi eq, %[[SELECT_VALUE]], %[[CST2]] : i32
// CHECK:         cf.cond_br %[[CMP_EQ]], ^[[BB_POINT2:bb[0-9]+]], ^bb2
// CHECK-LABEL: ^bb2:
// Check for the lower bound for the interval `case (4:5)`
// CHECK:        %[[CMP_SLE:.*]] = arith.cmpi sle, %[[CST4]], %[[SELECT_VALUE]] : i32
// CHECK:        cf.cond_br %[[CMP_SLE]], ^[[BB_UPPERBOUND5:bb[0-9]+]], ^bb4
// CHECK:       ^[[BB_UPPERBOUND5]]:
// Check for the upper bound for the interval `case (4:5)`
// CHECK:        %[[CMP_SLE:.*]] = arith.cmpi sle, %[[SELECT_VALUE]], %[[CST5]] : i32
// CHECK:        cf.cond_br %[[CMP_SLE]], ^[[BB_INTERVAL45:bb[0-9]+]], ^bb4
// CHECK-LABEL: ^bb4:
// Check for the point value 7 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_EQ:.*]] = arith.cmpi eq, %[[SELECT_VALUE]], %[[CST7]] : i32
// CHECK:        cf.cond_br %[[CMP_EQ]], ^[[BB_7_8_15_21:bb[0-9]+]], ^bb5
// CHECK-LABEL: ^bb5:
// Check for lower bound 8 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]] = arith.cmpi sle, %[[CST8]], %[[SELECT_VALUE]] : i32
// CHECK:        cf.cond_br %[[CMP_SLE]], ^[[BB_INTERVAL8_15:bb[0-9]+]], ^bb7
// CHECK:       ^[[BB_INTERVAL8_15]]:
// Check for upper bound 15 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]] = arith.cmpi sle, %[[SELECT_VALUE]], %[[CST15]] : i32
// CHECK:        cf.cond_br %[[CMP_SLE]], ^[[BB_7_8_15_21]], ^bb7
// CHECK-LABEL: ^bb7:
// Check for lower bound 21 in `case (7,8:15,21:)`
// CHECK:        %[[CMP_SLE:.*]]  = arith.cmpi sle, %[[CST21]], %[[SELECT_VALUE]] : i32
// CHECK:        cf.cond_br %[[CMP_SLE]], ^[[BB_7_8_15_21]], ^bb8
// CHECK-LABEL: ^bb8:
// Default branch (unit case) falls through to the CLASS DEFAULT successor.
// CHECK:         cf.br ^[[BB_DEFAULT:bb[0-9]+]]
// Each original case successor is preserved in source order after the
// ladder; each stores its case's constant then falls into the shared
// join block. The block names captured above are reused so the CFG
// topology is checked, not just the shape.
// CHECK:       ^[[BB_UPPER]]:
// CHECK:         %[[STORE_UPPER:.*]] = arith.constant 1 : i32
// CHECK:         fir.store %[[STORE_UPPER]] to %[[ARG0]] : !fir.ref<i32>
// CHECK:         cf.br ^[[BB_JOIN:bb[0-9]+]]
// CHECK:       ^[[BB_POINT2]]:
// CHECK:         %[[STORE_POINT2:.*]] = arith.constant 2 : i32
// CHECK:         fir.store %[[STORE_POINT2]] to %[[ARG0]] : !fir.ref<i32>
// CHECK:         cf.br ^[[BB_JOIN]]
// CHECK:       ^[[BB_DEFAULT]]:
// CHECK:         %[[STORE_DEFAULT:.*]] = arith.constant 0 : i32
// CHECK:         fir.store %[[STORE_DEFAULT]] to %[[ARG0]] : !fir.ref<i32>
// CHECK:         cf.br ^[[BB_JOIN]]
// CHECK:       ^[[BB_INTERVAL45]]:
// CHECK:         %[[STORE_INTERVAL45:.*]] = arith.constant 4 : i32
// CHECK:         fir.store %[[STORE_INTERVAL45]] to %[[ARG0]] : !fir.ref<i32>
// CHECK:         cf.br ^[[BB_JOIN]]
// CHECK:       ^[[BB_7_8_15_21]]:
// CHECK:         %[[STORE_7_8_15_21:.*]] = arith.constant 7 : i32
// CHECK:         fir.store %[[STORE_7_8_15_21]] to %[[ARG0]] : !fir.ref<i32>
// CHECK:         cf.br ^[[BB_JOIN]]
// Join block reads the ref and returns it.
// CHECK:       ^[[BB_JOIN]]:
// CHECK:         %[[RET:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
// CHECK:         return %[[RET]] : i32

// -----

// Exercises fir.select_case with an i1 selector (converted from a Fortran
// LOGICAL) and two #fir.point cases plus a unit default.

func.func @select_case_logical(%arg0: !fir.ref<!fir.logical<4>>) {
  %1 = fir.load %arg0 : !fir.ref<!fir.logical<4>>
  %2 = fir.convert %1 : (!fir.logical<4>) -> i1
  %false = arith.constant false
  %true = arith.constant true
  fir.select_case %2 : i1 [#fir.point, %false, ^bb1,
                           #fir.point, %true, ^bb2,
                            unit, ^bb3]
^bb1:
  %c1_i32 = arith.constant 1 : i32
  cf.br ^bb3
^bb2:
  %c2_i32 = arith.constant 2 : i32
  cf.br ^bb3
^bb3:
  return
}

// CHECK-LABEL: func.func @select_case_logical(
// CHECK-SAME:                          %[[ARG0:.*]]: !fir.ref<!fir.logical<4>>)
// CHECK:         %[[LD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.logical<4>>
// CHECK:         %[[SEL:.*]] = fir.convert %[[LD]] : (!fir.logical<4>) -> i1
// CHECK-DAG:     %[[FALSE:.*]] = arith.constant false
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK:         %[[C0:.*]] = arith.cmpi eq, %[[SEL]], %[[FALSE]] : i1
// CHECK:         cf.cond_br %[[C0]], ^[[BB_FALSE:bb[0-9]+]], ^bb1
// CHECK-LABEL: ^bb1:
// CHECK:         %[[C1:.*]] = arith.cmpi eq, %[[SEL]], %[[TRUE]] : i1
// CHECK:         cf.cond_br %[[C1]], ^[[BB_TRUE:bb[0-9]+]], ^bb2
// CHECK-LABEL: ^bb2:
// Unit-default fall-through.
// CHECK:         cf.br ^[[BB_JOIN:bb[0-9]+]]
// Two #fir.point successor bodies (each declares its constant then
// falls into the shared join block), and the join block returns.
// CHECK:       ^[[BB_FALSE]]:
// CHECK:         arith.constant 1 : i32
// CHECK:         cf.br ^[[BB_JOIN]]
// CHECK:       ^[[BB_TRUE]]:
// CHECK:         arith.constant 2 : i32
// CHECK:         cf.br ^[[BB_JOIN]]
// CHECK:       ^[[BB_JOIN]]:
// CHECK:         return

// -----

// Exercises fir.select_case with per-successor block operands: two
// #fir.point cases pass different `!fir.ref<!fir.array<2xi32>>` values to
// the same target block. The pass propagates the operand forwarding
// verbatim; block-argument types are preserved (no type conversion runs).

func.func @select_case_block_args(%arg0: !fir.ref<i32>, %arg1: !fir.ref<!fir.array<2xi32>>, %arg2: !fir.ref<!fir.array<2xi32>>) {
  %0 = fir.load %arg0 : !fir.ref<i32>
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  fir.select_case %0 : i32 [#fir.point, %c1, ^bb1(%arg1 : !fir.ref<!fir.array<2xi32>>),
                            #fir.point, %c2, ^bb1(%arg2 : !fir.ref<!fir.array<2xi32>>),
                            unit, ^bb2]
^bb1(%1: !fir.ref<!fir.array<2xi32>>):
  cf.br ^bb2
^bb2:
  return
}

// CHECK-LABEL: func.func @select_case_block_args
// CHECK:         cf.cond_br %{{.*}}, ^{{.*}}(%{{.*}} : !fir.ref<!fir.array<2xi32>>), ^{{.*}}
// CHECK:       ^{{.*}}(%{{.*}}: !fir.ref<!fir.array<2xi32>>):
