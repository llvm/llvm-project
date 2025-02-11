// Test hlfir.reshape operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

// reshape(x, y)
func.func @_QPreshape1(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 {uniq_name = "_QFreshape1Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %4 = hlfir.reshape %1#0 %3#0 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?xf32>
  hlfir.assign %4 to %1#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %4 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPreshape1(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFreshape1Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_9]]) dummy_scope %[[VAL_7]] {uniq_name = "_QFreshape1Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_10]]#1(%[[VAL_11]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_13:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_14:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_15:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.store %[[VAL_17]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_14]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_25]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_27:.*]] = fir.box_addr %[[VAL_25]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_28:.*]] = fir.shape_shift %[[VAL_26]]#0, %[[VAL_26]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_27]](%[[VAL_28]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
// CHECK:           %[[VAL_30:.*]] = hlfir.as_expr %[[VAL_29]]#0 move %[[VAL_2]] : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
// CHECK:           hlfir.assign %[[VAL_30]] to %[[VAL_8]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:           hlfir.destroy %[[VAL_30]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }

// reshape(x, y, pad)
func.func @_QPreshape2(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "pad"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFreshape2Epad"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape2Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFreshape2Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %5 = hlfir.reshape %2#0 %4#0 pad %1#0 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?xf32>>) -> !hlfir.expr<?xf32>
  hlfir.assign %5 to %2#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %5 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPreshape2(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"},
// CHECK-SAME:                           %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "pad"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape2Epad"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape2Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_11]]) dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape2Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_13]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_15:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_15]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_4]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_26]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_29:.*]] = fir.shape_shift %[[VAL_27]]#0, %[[VAL_27]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_28]](%[[VAL_29]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
// CHECK:           %[[VAL_31:.*]] = hlfir.as_expr %[[VAL_30]]#0 move %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
// CHECK:           hlfir.assign %[[VAL_31]] to %[[VAL_10]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:           hlfir.destroy %[[VAL_31]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }

// reshape(x, y, order=order)
func.func @_QPreshape3(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}, %arg2: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "order"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFreshape3Eorder"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape3Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFreshape3Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %5 = hlfir.reshape %2#0 %4#0 order %1#0 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xf32>
  hlfir.assign %5 to %2#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %5 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPreshape3(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"},
// CHECK-SAME:                           %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "order"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape3Eorder"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape3Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_11]]) dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape3Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_13]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_15:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_15]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_4]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_26]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = fir.box_addr %[[VAL_26]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_29:.*]] = fir.shape_shift %[[VAL_27]]#0, %[[VAL_27]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_28]](%[[VAL_29]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
// CHECK:           %[[VAL_31:.*]] = hlfir.as_expr %[[VAL_30]]#0 move %[[VAL_3]] : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
// CHECK:           hlfir.assign %[[VAL_31]] to %[[VAL_10]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:           hlfir.destroy %[[VAL_31]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }

// reshape(x, y, pad, order)
func.func @_QPreshape4(%arg0: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}, %arg2: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "pad"}, %arg3: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "order"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg3 dummy_scope %0 {uniq_name = "_QFreshape4Eorder"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFreshape4Epad"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape4Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %4 = fir.shape %c1 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %arg1(%4) dummy_scope %0 {uniq_name = "_QFreshape4Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %6 = hlfir.reshape %3#0 %5#0 pad %2#0 order %1#0 : (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?xf32>
  hlfir.assign %6 to %3#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %6 : !hlfir.expr<?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPreshape4(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"},
// CHECK-SAME:                           %[[VAL_2:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "pad"},
// CHECK-SAME:                           %[[VAL_3:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "order"}) {
// CHECK:           %[[VAL_4:.*]] = arith.constant true
// CHECK:           %[[VAL_5:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           %[[VAL_9:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_3]] dummy_scope %[[VAL_9]] {uniq_name = "_QFreshape4Eorder"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_11:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_9]] {uniq_name = "_QFreshape4Epad"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_9]] {uniq_name = "_QFreshape4Ex"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_13]]) dummy_scope %[[VAL_9]] {uniq_name = "_QFreshape4Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_15:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_16:.*]] = fir.embox %[[VAL_14]]#1(%[[VAL_15]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_17:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
// CHECK:           fir.store %[[VAL_19]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_20:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_12]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_16]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_11]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_5]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_8]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK:           %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_27]], %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_29:.*]] = fir.box_addr %[[VAL_27]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
// CHECK:           %[[VAL_30:.*]] = fir.shape_shift %[[VAL_28]]#0, %[[VAL_28]]#1 : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_31:.*]]:2 = hlfir.declare %[[VAL_29]](%[[VAL_30]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.array<?xf32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
// CHECK:           %[[VAL_32:.*]] = hlfir.as_expr %[[VAL_31]]#0 move %[[VAL_4]] : (!fir.box<!fir.array<?xf32>>, i1) -> !hlfir.expr<?xf32>
// CHECK:           hlfir.assign %[[VAL_32]] to %[[VAL_12]]#0 : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
// CHECK:           hlfir.destroy %[[VAL_32]] : !hlfir.expr<?xf32>
// CHECK:           return
// CHECK:         }

// subroutine reshape5(x, y)
//   use types
//   integer :: y(1)
//   type(t) :: x(:)
//   x = reshape(x, y)
// end subroutine reshape5
func.func @_QPreshape5(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape5Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 {uniq_name = "_QFreshape5Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %4 = hlfir.reshape %1#0 %3#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
  hlfir.assign %4 to %1#0 : !hlfir.expr<?x!fir.type<_QMtypesTt>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
  hlfir.destroy %4 : !hlfir.expr<?x!fir.type<_QMtypesTt>>
  return
}
// CHECK-LABEL:   func.func @_QPreshape5(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFreshape5Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_9]]) dummy_scope %[[VAL_7]] {uniq_name = "_QFreshape5Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_10]]#1(%[[VAL_11]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_13:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_14:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_15:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_15]](%[[VAL_16]]) : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_17]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_18:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !fir.box<none>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_14]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_18]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_19]], %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_25:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_26:.*]]:3 = fir.box_dims %[[VAL_25]], %[[VAL_4]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_27:.*]] = fir.shift %[[VAL_26]]#0 : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_28:.*]]:2 = hlfir.declare %[[VAL_25]](%[[VAL_27]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.shift<1>) -> (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>)
// CHECK:           %[[VAL_29:.*]] = hlfir.as_expr %[[VAL_28]]#0 move %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
// CHECK:           hlfir.assign %[[VAL_29]] to %[[VAL_8]]#0 : !hlfir.expr<?x!fir.type<_QMtypesTt>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           hlfir.destroy %[[VAL_29]] : !hlfir.expr<?x!fir.type<_QMtypesTt>>
// CHECK:           return
// CHECK:         }

// subroutine reshape6(x, y)
//   use types
//   integer :: y(1)
//   class(t), allocatable :: x(:)
//   x = reshape(x, y)
// end subroutine reshape6
func.func @_QPreshape6(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFreshape6Ex"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
  %2 = fir.shape %c1 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %arg1(%2) dummy_scope %0 {uniq_name = "_QFreshape6Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %4 = fir.load %1#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  %5 = hlfir.reshape %4 %3#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.ref<!fir.array<1xi32>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  hlfir.assign %5 to %1#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  hlfir.destroy %5 : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  return
}
// CHECK-LABEL:   func.func @_QPreshape6(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_3:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_7]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFreshape6Ex"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_9]]) dummy_scope %[[VAL_7]] {uniq_name = "_QFreshape6Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_8]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_10]]#1(%[[VAL_12]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_14:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_15:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) source_box %[[VAL_11]] : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.shape<1>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_6]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_11]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_14]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_15]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_3]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_6]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_26]], %[[VAL_4]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = fir.shift %[[VAL_27]]#0 : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_26]](%[[VAL_28]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.shift<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>)
// CHECK:           %[[VAL_30:.*]] = hlfir.as_expr %[[VAL_29]]#0 move %[[VAL_2]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
// CHECK:           hlfir.assign %[[VAL_30]] to %[[VAL_8]]#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           hlfir.destroy %[[VAL_30]] : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
// CHECK:           return
// CHECK:         }

// subroutine reshape7(x, y, pad)
//   use types
//   integer :: y(1)
//   type(t) :: x(:), pad(:)
//   x = reshape(x, y, pad)
// end subroutine reshape7
func.func @_QPreshape7(%arg0: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}, %arg2: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "pad"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFreshape7Epad"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreshape7Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFreshape7Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %5 = hlfir.reshape %2#0 %4#0 pad %1#0 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.ref<!fir.array<1xi32>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
  hlfir.assign %5 to %2#0 : !hlfir.expr<?x!fir.type<_QMtypesTt>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
  hlfir.destroy %5 : !hlfir.expr<?x!fir.type<_QMtypesTt>>
  return
}
// CHECK-LABEL:   func.func @_QPreshape7(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"},
// CHECK-SAME:                           %[[VAL_2:.*]]: !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "pad"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape7Epad"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape7Ex"} : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_11]]) dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape7Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_13]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_15:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_19:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !fir.box<none>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_14]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_15]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_19]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_4]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_26:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_27:.*]]:3 = fir.box_dims %[[VAL_26]], %[[VAL_5]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_28:.*]] = fir.shift %[[VAL_27]]#0 : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_29:.*]]:2 = hlfir.declare %[[VAL_26]](%[[VAL_28]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.shift<1>) -> (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>)
// CHECK:           %[[VAL_30:.*]] = hlfir.as_expr %[[VAL_29]]#0 move %[[VAL_3]] : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMtypesTt>>
// CHECK:           hlfir.assign %[[VAL_30]] to %[[VAL_10]]#0 : !hlfir.expr<?x!fir.type<_QMtypesTt>>, !fir.box<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           hlfir.destroy %[[VAL_30]] : !hlfir.expr<?x!fir.type<_QMtypesTt>>
// CHECK:           return
// CHECK:         }

// subroutine reshape8(x, y, pad)
//   use types
//   integer :: y(1)
//   class(t), allocatable :: x(:)
//   class(t) :: pad(:)
//   x = reshape(x, y, pad)
// end subroutine reshape8
func.func @_QPreshape8(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>> {fir.bindc_name = "x"}, %arg1: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"}, %arg2: !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "pad"}) {
  %c1 = arith.constant 1 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFreshape8Epad"} : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>)
  %2:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFreshape8Ex"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
  %3 = fir.shape %c1 : (index) -> !fir.shape<1>
  %4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFreshape8Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
  %5 = fir.load %2#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  %6 = hlfir.reshape %5 %4#0 pad %1#0 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.ref<!fir.array<1xi32>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  hlfir.assign %6 to %2#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  hlfir.destroy %6 : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  return
}
// CHECK-LABEL:   func.func @_QPreshape8(
// CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>> {fir.bindc_name = "x"},
// CHECK-SAME:                           %[[VAL_1:.*]]: !fir.ref<!fir.array<1xi32>> {fir.bindc_name = "y"},
// CHECK-SAME:                           %[[VAL_2:.*]]: !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>> {fir.bindc_name = "pad"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant {{[0-9]*}} : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_2]] dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape8Epad"} : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.dscope) -> (!fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFreshape8Ex"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_11]]) dummy_scope %[[VAL_8]] {uniq_name = "_QFreshape8Ey"} : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>, !fir.dscope) -> (!fir.ref<!fir.array<1xi32>>, !fir.ref<!fir.array<1xi32>>)
// CHECK:           %[[VAL_13:.*]] = fir.load %[[VAL_10]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_14]]) : (!fir.ref<!fir.array<1xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<1xi32>>
// CHECK:           %[[VAL_16:.*]] = fir.absent !fir.box<i1>
// CHECK:           %[[VAL_17:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_18:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_19:.*]] = fir.embox %[[VAL_17]](%[[VAL_18]]) source_box %[[VAL_13]] : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.shape<1>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_19]] to %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_20:.*]] = fir.address_of(@_QQcl
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_13]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<1xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.class<!fir.array<?x!fir.type<_QMtypesTt>>>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_16]] : (!fir.box<i1>) -> !fir.box<none>
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<!fir.char<1,{{[0-9]*}}>>) -> !fir.ref<i8>
// CHECK:           fir.call @_FortranAReshape(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[VAL_4]]) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.box<none>, !fir.ref<i8>, i32) -> ()
// CHECK:           %[[VAL_27:.*]] = fir.load %[[VAL_7]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[VAL_28:.*]]:3 = fir.box_dims %[[VAL_27]], %[[VAL_5]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_29:.*]] = fir.shift %[[VAL_28]]#0 : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_30:.*]]:2 = hlfir.declare %[[VAL_27]](%[[VAL_29]]) {uniq_name = ".tmp.intrinsic_result"} : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.shift<1>) -> (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>)
// CHECK:           %[[VAL_31:.*]] = hlfir.as_expr %[[VAL_30]]#0 move %[[VAL_3]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i1) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
// CHECK:           hlfir.assign %[[VAL_31]] to %[[VAL_10]]#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           hlfir.destroy %[[VAL_31]] : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
// CHECK:           return
// CHECK:         }
