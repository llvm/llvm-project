// Test hlfir.maxloc operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

// simple one argument maxloc
func.func @_QPmaxloc1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxloc1Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxloc1Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2 = hlfir.maxloc %0#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xi32>
  hlfir.assign %2 to %1#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %2 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc1(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFmaxloc1Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFmaxloc1Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %[[V3:.*]] = fir.absent !fir.box<i1>
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V5:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V6:.*]] = fir.embox %[[V4]](%[[V5]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V6]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V8:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V9:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:         %[[V12:.*]] = fir.convert %[[V3]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK-NEXT:    %[[V13:.*]] = fir.call @_FortranAMaxlocInteger4(%[[V8]], %[[V9]], %c4_i32, {{.*}}, {{.*}}, %[[V12]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V14:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V15:.*]]:3 = fir.box_dims %[[V14]], %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V16:.*]] = fir.box_addr %[[V14]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V17:.*]] = fir.shape_shift %[[V15]]#0, %[[V15]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V18:.*]]:2 = hlfir.declare %[[V16]](%[[V17]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %[[V19:.*]] = hlfir.as_expr %[[V18]]#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V19]] to %[[V2]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V19]] : !hlfir.expr<?xi32>


// maxloc with a dim
func.func @_QPmaxloc2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.ref<index> {fir.bindc_name = "d"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxloc2Ea"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxloc2Ed"} : (!fir.ref<index>) -> (!fir.ref<index>, !fir.ref<index>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxloc2Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.load %1#0 : !fir.ref<index>
  %4 = hlfir.maxloc %0#0 dim %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x?xi32>>, index) -> !hlfir.expr<?xi32>
  hlfir.assign %4 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %4 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc2(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK:           %[[ARG2:.*]]: !fir.ref<index>
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFmaxloc2Ea"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFmaxloc2Ed"} : (!fir.ref<index>) -> (!fir.ref<index>, !fir.ref<index>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFmaxloc2Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V4:.*]] = fir.load %[[V2]]#0 : !fir.ref<index>
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %[[V5:.*]] = fir.convert %[[V4]] : (index) -> i32
// CHECK-NEXT:    %[[V6:.*]] = fir.absent !fir.box<i1>
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V7:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V8:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V9:.*]] = fir.embox %[[V7]](%[[V8]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V9]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V11:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V12:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:         %[[V15:.*]] = fir.convert %[[V6]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK-NEXT:    %[[V16:.*]] = fir.call @_FortranAMaxlocDim(%[[V11]], %[[V12]], %c4_i32, %[[V5]], {{.*}}, {{.*}}, %[[V15]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V17:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V18:.*]]:3 = fir.box_dims %[[V17]], %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V19:.*]] = fir.box_addr %[[V17]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V20:.*]] = fir.shape_shift %[[V18]]#0, %[[V18]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V21:.*]]:2 = hlfir.declare %[[V19]](%[[V20]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %[[V22:.*]] = hlfir.as_expr %[[V21]]#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V22]] to %[[V3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V22]] : !hlfir.expr<?xi32>
// CHECK-NEXT:    return


// maxloc with scalar mask
func.func @_QPmaxloc3(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.ref<!fir.logical<4>> {fir.bindc_name = "m"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxloc3Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxloc3Em"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxloc3Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = hlfir.maxloc %0#0 mask %1#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.logical<4>>) -> !hlfir.expr<?xi32>
  hlfir.assign %3 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %3 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc3(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK:           %[[ARG2:.*]]: !fir.ref<!fir.logical<4>>
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFmaxloc3Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFmaxloc3Em"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFmaxloc3Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %[[V4:.*]] = fir.embox %[[V2]]#1 : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V5:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V6:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V7:.*]] = fir.embox %[[V5]](%[[V6]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V7]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V9:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V10:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:         %[[V13:.*]] = fir.convert %[[V4]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK-NEXT:    %[[V14:.*]] = fir.call @_FortranAMaxlocInteger4(%[[V9]], %[[V10]], %c4_i32, {{.*}}, {{.*}}, %[[V13]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V15:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V16:.*]]:3 = fir.box_dims %[[V15]], %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V17:.*]] = fir.box_addr %[[V15]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V18:.*]] = fir.shape_shift %[[V16]]#0, %[[V16]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V19:.*]]:2 = hlfir.declare %[[V17]](%[[V18]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %[[V20:.*]] = hlfir.as_expr %[[V19]]#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V20]] to %[[V3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V20]] : !hlfir.expr<?xi32>
// CHECK-NEXT:    return


// maxloc with array mask
func.func @_QPmaxloc4(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxloc4Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg2 {uniq_name = "_QFmaxloc4Em"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxloc4Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = hlfir.maxloc %0#0 mask %1#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?x!fir.logical<4>>>) -> !hlfir.expr<?xi32>
  hlfir.assign %3 to %2#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %3 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc4(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK:           %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFmaxloc4Ea"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFmaxloc4Em"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFmaxloc4Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V5:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V6:.*]] = fir.embox %[[V4]](%[[V5]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V6]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V8:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V9:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:         %[[V12:.*]] = fir.convert %[[V2]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK-NEXT:    %[[V13:.*]] = fir.call @_FortranAMaxlocInteger4(%[[V8]], %[[V9]], %c4_i32, {{.*}}, {{.*}}, %[[V12]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V14:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V15:.*]]:3 = fir.box_dims %[[V14]], %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V16:.*]] = fir.box_addr %[[V14]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V17:.*]] = fir.shape_shift %[[V15]]#0, %[[V15]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V18:.*]]:2 = hlfir.declare %[[V16]](%[[V17]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %[[V19:.*]] = hlfir.as_expr %[[V18]]#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V19]] to %[[V3]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V19]] : !hlfir.expr<?xi32>
// CHECK-NEXT:    return


fir.global internal @_QFmaxloc5Ea : !fir.array<2x2xi32> {
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

// 3 argument maxloc, using local variables
func.func @_QPmaxloc5(%arg0: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "s"}) {
  %0 = fir.address_of(@_QFmaxloc5Ea) : !fir.ref<!fir.array<2x2xi32>>
  %c2 = arith.constant 2 : index
  %c2_0 = arith.constant 2 : index
  %1 = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFmaxloc5Ea"} : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xi32>>, !fir.ref<!fir.array<2x2xi32>>)
  %c2_1 = arith.constant 2 : index
  %3 = fir.shape %c2_1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg0(%3) {uniq_name = "_QFmaxloc5Es"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %5 = hlfir.maxloc %2#0 dim %c1_i32 mask %true {fastmath = #arith.fastmath<contract>} : (!fir.ref<!fir.array<2x2xi32>>, i32, i1) -> !hlfir.expr<2xi32>
  hlfir.assign %5 to %4#0 : !hlfir.expr<2xi32>, !fir.ref<!fir.array<2xi32>>
  hlfir.destroy %5 : !hlfir.expr<2xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc5(
// CHECK:           %[[ARG0:.*]]: !fir.ref<!fir.array<2xi32>> {fir.bindc_name = "s"}
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:    %[[V2:.*]] = fir.address_of(@_QFmaxloc5Ea) : !fir.ref<!fir.array<2x2xi32>>
// CHECK-NEXT:    %c2 = arith.constant 2 : index
// CHECK-NEXT:    %c2_0 = arith.constant 2 : index
// CHECK-NEXT:    %[[V3:.*]] = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:    %[[V4:.*]]:2 = hlfir.declare %[[V2]](%[[V3]]) {uniq_name = "_QFmaxloc5Ea"} : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xi32>>, !fir.ref<!fir.array<2x2xi32>>)
// CHECK-NEXT:    %c2_1 = arith.constant 2 : index
// CHECK-NEXT:    %[[V5:.*]] = fir.shape %c2_1 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V6:.*]]:2 = hlfir.declare %[[ARG0]](%[[V5]]) {uniq_name = "_QFmaxloc5Es"} : (!fir.ref<!fir.array<2xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<2xi32>>, !fir.ref<!fir.array<2xi32>>)
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %[[V7:.*]] = fir.shape %c2, %c2_0 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:    %[[V8:.*]] = fir.embox %[[V4]]#1(%[[V7]]) : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<2x2xi32>>
// CHECK-NEXT:    %[[V9:.*]] = fir.convert %true : (i1) -> !fir.logical<4>
// CHECK-NEXT:    fir.store %[[V9]] to %[[V1]] : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:    %[[V10:.*]] = fir.embox %[[V1]] : (!fir.ref<!fir.logical<4>>) -> !fir.box<!fir.logical<4>>
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V11:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V12:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V13:.*]] = fir.embox %[[V11]](%[[V12]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V13]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V15:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V16:.*]] = fir.convert %[[V8]] : (!fir.box<!fir.array<2x2xi32>>) -> !fir.box<none>
// CHECK:         %[[V19:.*]] = fir.convert %[[V10]] : (!fir.box<!fir.logical<4>>) -> !fir.box<none>
// CHECK-NEXT:    %[[V20:.*]] = fir.call @_FortranAMaxlocDim(%[[V15]], %[[V16]], %c4_i32, %c1_i32, {{.*}}, {{.*}}, %[[V19]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V21:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_2 = arith.constant 0 : index
// CHECK-NEXT:    %[[V22:.*]]:3 = fir.box_dims %[[V21]], %c0_2 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V23:.*]] = fir.box_addr %[[V21]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V24:.*]] = fir.shape_shift %[[V22]]#0, %[[V22]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V25:.*]]:2 = hlfir.declare %[[V23]](%[[V24]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true_3 = arith.constant true
// CHECK-NEXT:    %[[V26:.*]] = hlfir.as_expr %[[V25]]#0 move %true_3 : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V26]] to %[[V6]]#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<2xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V26]] : !hlfir.expr<?xi32>
// CHECK-NEXT:    return


// simple one argument maxloc for character
func.func @_QPmaxloc6(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaxloc6Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "_QFmaxloc4Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2 = hlfir.maxloc %0#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !hlfir.expr<?xi32>
  hlfir.assign %2 to %1#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  hlfir.destroy %2 : !hlfir.expr<?xi32>
  return
}
// CHECK-LABEL: func.func @_QPmaxloc6(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFmaxloc6Ea"} : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.box<!fir.array<?x!fir.char<1,?>>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFmaxloc4Es"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %[[V3:.*]] = fir.absent !fir.box<i1>
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V5:.*]] = fir.shape %c0 : (index) -> !fir.shape<1>
// CHECK-NEXT:    %[[V6:.*]] = fir.embox %[[V4]](%[[V5]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK-NEXT:    fir.store %[[V6]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:         %[[V8:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V9:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<none>
// CHECK:         %[[V12:.*]] = fir.convert %[[V3]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK-NEXT:    %[[V13:.*]] = fir.call @_FortranAMaxlocCharacter(%[[V8]], %[[V9]], %c4_i32, {{.*}}, {{.*}}, %[[V12]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V14:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK-NEXT:    %c0_0 = arith.constant 0 : index
// CHECK-NEXT:    %[[V15:.*]]:3 = fir.box_dims %[[V14]], %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK-NEXT:    %[[V16:.*]] = fir.box_addr %[[V14]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
// CHECK-NEXT:    %[[V17:.*]] = fir.shape_shift %[[V15]]#0, %[[V15]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK-NEXT:    %[[V18:.*]]:2 = hlfir.declare %[[V16]](%[[V17]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
// CHECK-NEXT:    %true = arith.constant true
// CHECK-NEXT:    %[[V19:.*]] = hlfir.as_expr %[[V18]]#0 move %true : (!fir.box<!fir.array<?xi32>>, i1) -> !hlfir.expr<?xi32>
// CHECK-NEXT:    hlfir.assign %[[V19]] to %[[V2]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    hlfir.destroy %[[V19]] : !hlfir.expr<?xi32>
// CHECK-NEXT:    return


// including mask and back
func.func @_QPmaxloc7(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "d"}, %arg2: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}, %arg3: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}, %arg4: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}) {
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFFtestEa"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
    %1:2 = hlfir.declare %arg3 {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestEd"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEm"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
    %4:2 = hlfir.declare %arg4 {uniq_name = "_QFFtestEs"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
    %5 = fir.load %2#0 : !fir.ref<i32>
    %6 = hlfir.maxloc %0#0 dim %5 mask %3#0 {fastmath = #arith.fastmath<contract>} : (!fir.box<!fir.array<?xi32>>, i32, !fir.box<!fir.array<?x!fir.logical<4>>>) -> i32
    hlfir.assign %6 to %4#0 : i32, !fir.box<!fir.array<?xi32>>
    return
}
// CHECK-LABEL: func.func @_QPmaxloc7(
// CHECK:           %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
// CHECK:           %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "d"}
// CHECK:           %[[ARG2:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "m"}
// CHECK:           %[[ARG3:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "b"}
// CHECK:           %[[ARG4:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "s"}
// CHECK-NEXT:    %[[V0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
// CHECK-NEXT:    %[[V1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFFtestEa"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V2:.*]]:2 = hlfir.declare %[[ARG3]] {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
// CHECK-NEXT:    %[[V3:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFFtestEd"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK-NEXT:    %[[V4:.*]]:2 = hlfir.declare %[[ARG2]] {uniq_name = "_QFFtestEm"} : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
// CHECK-NEXT:    %[[V5:.*]]:2 = hlfir.declare %[[ARG4]] {uniq_name = "_QFFtestEs"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK-NEXT:    %[[V6:.*]] = fir.load %[[V3]]#0 : !fir.ref<i32>
// CHECK-NEXT:    %c4_i32 = arith.constant 4 : i32
// CHECK-NEXT:    %false = arith.constant false
// CHECK-NEXT:    %[[V7:.*]] = fir.zero_bits !fir.heap<i32>
// CHECK-NEXT:    %[[V8:.*]] = fir.embox %[[V7]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
// CHECK-NEXT:    fir.store %[[V8]] to %[[V0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK:         %[[V10:.*]] = fir.convert %[[V0]] : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<none>>
// CHECK-NEXT:    %[[V11:.*]] = fir.convert %[[V1]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:         %[[V14:.*]] = fir.convert %[[V4]]#1 : (!fir.box<!fir.array<?x!fir.logical<4>>>) -> !fir.box<none>
// CHECK-NEXT:    %[[V15:.*]] = fir.call @_FortranAMaxlocDim(%[[V10]], %[[V11]], %c4_i32, %[[V6]], {{.*}}, {{.*}}, %[[V14]], %false) fastmath<contract> : (!fir.ref<!fir.box<none>>, !fir.box<none>, i32, i32, !fir.ref<i8>, i32, !fir.box<none>, i1) -> none
// CHECK-NEXT:    %[[V16:.*]] = fir.load %[[V0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK-NEXT:    %[[V17:.*]] = fir.box_addr %[[V16]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK-NEXT:    %[[V18:.*]] = fir.load %[[V17]] : !fir.heap<i32>
// CHECK-NEXT:    fir.freemem %[[V17]] : !fir.heap<i32>
// CHECK-NEXT:    hlfir.assign %[[V18]] to %[[V5]]#0 : i32, !fir.box<!fir.array<?xi32>>
// CHECK-NEXT:    return

