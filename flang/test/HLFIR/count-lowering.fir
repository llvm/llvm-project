// Test hlfir.count operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

func.func @_QPcount1(%arg0: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFcount1Ea"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFcount1Es"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = hlfir.count %0#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  hlfir.assign %2 to %1#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPcount1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:         %[[RET_ARG:.*]] = fir.call @_FortranACount(%[[MASK_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[C1:.*]]) : (!fir.box<none>, !fir.ref<i8>, i32, i32) -> i64
// CHECK-NEXT:    %[[RET:.*]] = fir.convert %[[RET_ARG]] : (i64) -> i32
// CHECK-NEXT:    hlfir.assign %[[RET]] to %[[RES]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPcount2(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.ref<i32> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFcount2Ea"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFcount2Ed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFcount2Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = hlfir.count %0#0 dim %3 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?xi32>
  hlfir.assign %4 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %4 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPcount2(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>
// CHECK:           %[[ARG2:.*]]: !fir.ref<i32>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[KIND:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[DIM:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<i32>
// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1

// CHECK:         %[[NONE:.*]] = fir.call @_FortranACountDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[KIND]], %[[LOC_STR:.*]], %[[LOC_N:.*]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK:         hlfir.assign %[[EXPR]] to %[[RES]]#0
// CHECK:         hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPcount3(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "s"}) {
  %0 = fir.address_of(@_QFcount3Ea) : !fir.ref<!fir.array<2x2x!fir.logical<4>>>
  %c2 = arith.constant 2 : index
  %c2_0 = arith.constant 2 : index
  %1 = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFcount3Ea"} : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, !fir.ref<!fir.array<2x2x!fir.logical<4>>>)
  %c2_1 = arith.constant 2 : index
  %3 = fir.shape %c2_1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg0(%3) {uniq_name = "_QFcount3Es"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %c1_i32 = arith.constant 1 : i32
  %5 = hlfir.count %2#0 dim %c1_i32 {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2x!fir.logical<4>>>, i32) -> !hlfir.expr<2xi32>
  hlfir.assign %5 to %4#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
  hlfir.destroy %5 : !hlfir.expr<2xi32>
  return
}
// CHECK-LABEL:  func.func @_QPcount3(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-DAG:     %[[KIND:.*]] = arith.constant 4 : i32
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
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

// CHECK:         %[[NONE:.*]] = fir.call @_FortranACountDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[KIND]], %[[LOC_STR:.*]], %[[LOC_N:.*]])
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK:         hlfir.assign %[[EXPR]] to %[[RES]]
// CHECK:         hlfir.destroy %[[EXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

func.func @_QPcount4(%arg0: !fir.box<!fir.array<?x?x!fir.logical<4>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.ref<i32> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFcount4Ea"} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x?x!fir.logical<4>>>, !fir.box<!fir.array<?x?x!fir.logical<4>>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFcount4Ed"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFcount4Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.load %1#0 : !fir.ref<i32>
  %4 = hlfir.count %0#0 dim %3 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?x!fir.logical<4>>>, i32) -> !hlfir.expr<?xi64>
  %5 = hlfir.shape_of %4 : (!hlfir.expr<?xi64>) -> !fir.shape<1>
  %6 = hlfir.elemental %5 : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
  ^bb0(%arg3: index):
    %7 = hlfir.apply %4, %arg3 : (!hlfir.expr<?xi64>, index) -> i64
    %8 = fir.convert %7 : (i64) -> i32
    hlfir.yield_element %8 : i32
  }
  hlfir.assign %6 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %6 : !hlfir.expr<?xi32>
  hlfir.destroy %4 : !hlfir.expr<?xi64>
  return
}
// CHECK-LABEL: func.func @_QPcount4(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?x!fir.logical<4>>>
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>
// CHECK:           %[[ARG2:.*]]: !fir.ref<i32>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi64>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi64>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[DIM:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<i32>
// CHECK-DAG:     %[[KIND:.*]] = arith.constant 8 : i32
// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1

// CHECK:         %[[NONE:.*]] = fir.call @_FortranACountDim(%[[RET_ARG]], %[[MASK_ARG]], %[[DIM]], %[[KIND]], %[[LOC_STR:.*]], %[[LOC_N:.*]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32) -> none
// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
// CHECK-NEXT:    %[[OUT_SHAPE:.*]] = hlfir.shape_of %[[EXPR]]
// CHECK-NEXT:    %[[OUT:.*]] = hlfir.elemental %[[OUT_SHAPE]] : (!fir.shape<1>) -> !hlfir.expr<?xi32>
// CHECK-DAG:       hlfir.assign %[[OUT]] to %[[RES]]#0
// CHECK-NEXT:      hlfir.destroy %[[OUT]] : !hlfir.expr<?xi32>
// CHECK-NEXT:      hlfir.destroy %[[EXPR]] : !hlfir.expr<?xi64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
