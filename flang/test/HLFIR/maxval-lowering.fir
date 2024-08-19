// Test hlfir.maxval operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

// simple one argument maxval
func.func @_QPmaxval1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxval1Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxval1Es"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = hlfir.maxval %0#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> i32
  hlfir.assign %2 to %1#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPmaxval1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>
// CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
// CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[MASK:.*]] = fir.absent !fir.box<i1>
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:         %[[RET:.*]] = fir.call @_FortranAMaxvalInteger4(%[[ARRAY_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[INT:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
// CHECK-NEXT:    hlfir.assign %[[RET]] to %[[RES]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// maxval with a by-ref dimension of index type
func.func @_QPmaxval2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.ref<index> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxval2Ea"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxval2Ed"} : (!fir.ref<index>) -> (!fir.ref<index>, !fir.ref<index>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxval2Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.load %1#0 : !fir.ref<index>
  %4 = hlfir.maxval %0#0 dim %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
  hlfir.assign %4 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %4 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxval2(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>
// CHECK:           %[[ARG2:.*]]: !fir.ref<index>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[DIM_VAR:.*]]:2 = hlfir.declare %[[ARG2]]

// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[MASK:.*]] = fir.absent !fir.box<i1>
// CHECK-DAG:     %[[DIM_IDX:.*]] = fir.load %[[DIM_VAR]]#0 : !fir.ref<index>
// CHECK-DAG:     %[[DIM:.*]] = fir.convert %[[DIM_IDX]] : (index) -> i32

// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAMaxvalDim(%[[RET_ARG]], %[[ARRAY_ARG]], %[[DIM]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> none

// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
// TODO: fix alias analysis in hlfir.assign bufferization
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]](%[[SHIFT]]) {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[ASEXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK:         hlfir.assign %[[ASEXPR]] to %[[RES]]#0
// CHECK:         hlfir.destroy %[[ASEXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// maxval with scalar mask
func.func @_QPmaxval3(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "m"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxval3Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxval3Em"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxval3Es"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = hlfir.maxval %0#0 mask %1#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.logical<4>>) -> i32
  hlfir.assign %3 to %2#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPmaxval3(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>
// CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
// CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
// CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[MASK_BOX:.*]] = fir.embox %[[MASK]]#1 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK_BOX]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK:         %[[RET:.*]] = fir.call @_FortranAMaxvalInteger4(%[[ARRAY_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[INT:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
// CHECK-NEXT:    hlfir.assign %[[RET]] to %[[RES]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// maxval with array mask
func.func @_QPmaxval4(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "s"}, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxval4Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxval4Em"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxval4Es"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = hlfir.maxval %0#0 mask %1#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
  hlfir.assign %3 to %2#0 : i32, !fir.ref<i32>
  return
}
// CHECK-LABEL: func.func @_QPmaxval4(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>
// CHECK:           %[[ARG1:.*]]: !fir.ref<i32>
// CHECK:           %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[ARG1]]
// CHECK-DAG:     %[[MASK:.*]]:2 = hlfir.declare %[[ARG2]]
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK:         %[[RET:.*]] = fir.call @_FortranAMaxvalInteger4(%[[ARRAY_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[INT:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32, i32, !fir.box<none>) -> i32
// CHECK-NEXT:    hlfir.assign %[[RET]] to %[[RES]]#0 : i32, !fir.ref<i32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }

fir.global internal @_QFmaxval5Ea : !fir.array<2x2xi32> {
  %0 = fir.undefined !fir.array<2x2xi32>
  %c1_i32 = arith.constant 1 : i32
  %1 = fir.insert_value %0, %c1_i32, [0 : index, 0 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
  %c2_i32 = arith.constant 2 : i32
  %2 = fir.insert_value %1, %c2_i32, [1 : index, 0 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
  %c3_i32 = arith.constant 3 : i32
  %3 = fir.insert_value %2, %c3_i32, [0 : index, 1 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
  %c4_i32 = arith.constant 4 : i32
  %4 = fir.insert_value %3, %c4_i32, [1 : index, 1 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
  %c2 = arith.constant 2 : index
  %c2_0 = arith.constant 2 : index
  fir.has_value %4 : !fir.array<2x2xi32>
}

// 3 argument maxval, using local variables
func.func @_QPmaxval5(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "s"}) {
  %0 = fir.address_of(@_QFmaxval5Ea) : !fir.ref<!fir.array<2x2xi32>>
  %c2 = arith.constant 2 : index
  %c2_0 = arith.constant 2 : index
  %1 = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFmaxval5Ea"} : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xi32>>, !fir.ref<!fir.array<2x2xi32>>)
  %c2_1 = arith.constant 2 : index
  %3 = fir.shape %c2_1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg0(%3) {uniq_name = "_QFmaxval5Es"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %5 = hlfir.maxval %2#0 dim %c1_i32 mask %true {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2xi32>>, i32, i1) -> !hlfir.expr<2xi32>
  hlfir.assign %5 to %4#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
  hlfir.destroy %5 : !hlfir.expr<2xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxval5(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>>
// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_SHAPE:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]](%[[RET_SHAPE]])
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[RES_VAR:.*]] = hlfir.declare %[[ARG0]](%[[RES_SHAPE:.*]])

// CHECK-DAG:     %[[MASK_ALLOC:.*]] = fir.alloca !fir.logical<4>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[MASK_VAL:.*]]  = fir.convert %[[TRUE]] : (i1) -> !fir.logical<4>
// CHECK-DAG:     fir.store %[[MASK_VAL]] to %[[MASK_ALLOC]] : !fir.ref<!fir.logical<4>>
// CHECK-DAG:     %[[MASK_BOX:.*]] = fir.embox %[[MASK_ALLOC]]

// CHECK-DAG:     %[[ARRAY_ADDR:.*]] = fir.address_of
// CHECK-DAG:     %[[ARRAY_VAR:.*]]:2 = hlfir.declare %[[ARRAY_ADDR]](%[[ARRAY_SHAPE:.*]])
// CHECK-DAG:     %[[ARRAY_BOX:.*]] = fir.embox %[[ARRAY_VAR]]#1(%[[ARRAY_SHAPE:.*]])

// CHECK-DAG:     %[[DIM:.*]] = arith.constant 1 : i32

// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY_BOX]] : (!fir.box<!fir.array<2x2xi32>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK_BOX]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAMaxvalDim(%[[RET_ARG]], %[[ARRAY_ARG]], %[[DIM]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>) -> none

// simple one argument maxval for character
func.func @_QPmaxval6(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxval6Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  %1:2 = fir.unboxchar %arg1 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %2:2 = hlfir.declare %1#0 typeparams %1#1 {uniq_name = "_QFmaxval6Es"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  %3 = hlfir.maxval %0#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<!fir.char<1,?>>
  hlfir.assign %3 to %2#0 : !hlfir.expr<!fir.char<1,?>>, !fir.boxchar<1>
  hlfir.destroy %3 : !hlfir.expr<!fir.char<1,?>>
  return
}
// CHECK-LABEL: func.func @_QPmaxval6(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>
// CHECK:           %[[ARG1:.*]]: !fir.boxchar<1>
// CHECK-DAG:     %[[TRUE:.*]] = arith.constant true
// CHECK-DAG:     %[[ARRAY:.*]]:2 = hlfir.declare %[[ARG0]]
// CHECK-DAG:     %[[UNBOXED:.*]]:2 = fir.unboxchar %[[ARG1]]
// CHECK-DAG:     %[[RES:.*]]:2 = hlfir.declare %[[UNBOXED]]#0 typeparams %[[UNBOXED]]#1

// CHECK-DAG:     %[[MASK:.*]] = fir.absent !fir.box<i1>
// CHECK-DAG:     %[[RET_BOX:.*]] = fir.alloca !fir.box<!fir.heap<!fir.char<1,?>>>
// CHECK-DAG:     %[[RET_ADDR:.*]] = fir.zero_bits !fir.heap<!fir.char<1,?>>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[RET_EMBOX:.*]] = fir.embox %[[RET_ADDR]] typeparams %[[C0]]
// CHECK-DAG:     fir.store %[[RET_EMBOX]] to %[[RET_BOX]]

// CHECK-DAG:     %[[RET_ARG:.*]] = fir.convert %[[RET_BOX]]
// CHECK-DAG:     %[[ARRAY_ARG:.*]] = fir.convert %[[ARRAY]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK-DAG:     %[[MASK_ARG:.*]] = fir.convert %[[MASK]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:         %[[NONE:.*]] = fir.call @_FortranAMaxvalCharacter(%[[RET_ARG]], %[[ARRAY_ARG]], %[[LOC_STR:.*]], %[[LOC_N:.*]], %[[MASK_ARG]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.ref<i8>, i32, !fir.box<none>) -> none

// CHECK:         %[[RET:.*]] = fir.load %[[RET_BOX]]
// CHECK:         %[[BOX_ELESIZE:.*]] = fir.box_elesize %[[RET]]
// CHECK-NEXT:    %[[ADDR:.*]] = fir.box_addr %[[RET]]
// CHECK-NEXT:    %[[TMP:.*]]:2 = hlfir.declare %[[ADDR]] typeparams %[[BOX_ELESIZE]] {uniq_name = ".tmp.intrinsic_result"}
// CHECK:         %[[ASEXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move %[[TRUE]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
// CHECK:         hlfir.assign %[[ASEXPR]] to %[[RES]]#0
// CHECK:         hlfir.destroy %[[ASEXPR]]
// CHECK-NEXT:    return
// CHECK-NEXT:  }
