// Test hlfir.all operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPall1(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.logical<4>> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFall1Ea"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFall1Es"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2 = hlfir.all %0#0 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.logical<4>
  hlfir.assign %2 to %1#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @_QPall1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:         %[[RET_ARG:.*]] = fir.call @_FortranAAll(%[[MASK_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[C1:.*]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i1
// CHECK-NEXT:    %[[RET:.*]] = fir.convert %[[RET_ARG]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:    hlfir.assign %[[RET]] to %[[RES]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPall2(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "s"}, %arg2: !fir.ref<i32> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFall2Ea"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFall2Ed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFall2Es"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = hlfir.all %0#0 dim %3 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
  hlfir.assign %4 to %2#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
  hlfir.destroy %4 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL: func.func @_QPall2(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:           %[[ARG2:.*]]: !fir.ref<i32>
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[DIM:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<i32>
// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1

// CHECK:         %[[NONE:.*]] = fir.call @_FortranAAllDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[LOC_STR:.*]], %[[LOC_N:.*]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i1) -> !hlfir.expr<?x!fir.logical<4>>
// CHECK:         hlfir.assign %[[EXPR]] to %[[RES]]#0
// CHECK:         hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPall3(%arg0: !fir.ref<!fir.array<2x!fir.logical<4>>> {fir.bindc_name = "s"}) {
  %0 = fir.address_of(@_QFall3Ea) : !fir.ref<!fir.array<2x2x!fir.logical<4>>>
  %c2 = arith.constant 2 : index
  %c2_0 = arith.constant 2 : index
  %1 = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFall3Ea"} : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, !fir.ref<!fir.array<2x2x!fir.logical<4>>>)
  %c2_1 = arith.constant 2 : index
  %3 = fir.shape %c2_1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg0(%3) {uniq_name = "_QFall3Es"} : (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2x!fir.logical<4>>>, !fir.ref<!fir.array<2x!fir.logical<4>>>)
  %c1_i32 = arith.constant 1 : i32
  %5 = hlfir.all %2#0 dim %c1_i32 : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2x!fir.logical<4>>
  hlfir.assign %5 to %4#0 : !hlfir.expr<2x!fir.logical<4>>, !fir.ref<!fir.array<2x!fir.logical<4>>>
  hlfir.destroy %5 : !hlfir.expr<2x!fir.logical<4>>
  return
}
// CHECK-LABEL:  func.func @_QPall3(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2x!fir.logical<4>>>
// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG0]](%[[RES_SHAPE:.*]])

// CHECK-DAG:     %[[MASK_ADDR:.*]] = fir.address_of
// CHECK-DAG:     %[[MASK_VAR:.*]]:2 = hlfir.declare %[[MASK_ADDR]](%[[MASK_SHAPE:.*]])
// CHECK-DAG:     %[[MASK_BOX:.*]] = fir.embox %[[MASK_VAR]]#1(%[[MASK_SHAPE:.*]])

// CHECK-DAG:     %[[DIM:.*]] = arith.constant 1 : i32

// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK_BOX]] : (!fir.box<!fir.array<2x2x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAAllDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[LOC_STR:.*]], %[[LOC_N:.*]])
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i1) -> !hlfir.expr<?x!fir.logical<4>>
// CHECK:         hlfir.assign %[[EXPR]] to %[[RES]]
// CHECK:         hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPall4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "s"}, %arg2: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFall4Ea"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg2 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFall4Ed"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFall4Es"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %3 = fir.load %1#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
  %4 = fir.box_addr %3 : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
  %5 = fir.load %4 : !fir.ptr<i32>
  %6 = hlfir.no_reassoc %5 : i32
  %7 = hlfir.all %0#0 dim %6 : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?x!fir.logical<4>>
  hlfir.assign %7 to %2#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
  hlfir.destroy %7 : !hlfir.expr<?x!fir.logical<4>>
  return
}
// CHECK-LABEL:  func.func @_QPall4(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[DIM_ARG:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.logical<4>>>>
// CHECK-DAG:     %[[DIM_PTR:.*]] = fir.load %[[DIM_ARG]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
// CHECK-DAG:     %[[DIM_ADDR:.*]] = fir.box_addr %[[DIM_PTR]]
// CHECK-DAG:     %[[DIM_VAR:.*]] = fir.load %[[DIM_ADDR]]
// CHECK-DAG:     %[[DIM:.*]] = hlfir.no_reassoc %[[DIM_VAR]]

// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.logical<4>>>
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]
// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1

// CHECK:         %[[NONE:.*]] = fir.call @_FortranAAllDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[LOC_STR:.*]], %[[LOC_N:.*]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> none
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[TRUE:.*]] = arith.constant true
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, i1) -> !hlfir.expr<?x!fir.logical<4>>
// CHECK:         hlfir.assign %[[EXPR]] to %[[RES]]
// CHECK:         hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }