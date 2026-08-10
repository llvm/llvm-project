// Test hlfir.eoshift operation lowering to fir runtime call
// RUN: fir-opt %s -lower-hlfir-intrinsics | FileCheck %s

// 1d boxed vector shift by scalar
func.func @eoshift1(%arg0: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "sh"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "sh"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2 = hlfir.eoshift %0#0 %1#0 : (!fir.box<!fir.array<?xi32>>, !fir.ref<i32>) -> !hlfir.expr<?xi32>
  hlfir.assign %2 to %0#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
  return
}
// CHECK-LABEL:   func.func @eoshift1(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "sh"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "sh"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_8:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]] = fir.embox %[[VAL_8]](%[[VAL_9]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
// CHECK:           fir.store %[[VAL_10]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_6]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
// CHECK:           fir.call @_FortranAEoshiftVector(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[BOUNDARY]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.box<none>, !fir.ref<i8>, i32) -> ()

// 2d boxed array shift by scalar
func.func @eoshift2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: i32 {fir.bindc_name = "sh"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %2 = hlfir.eoshift %0#0 %arg1 : (!fir.box<!fir.array<?x?xi32>>, i32) -> !hlfir.expr<?x?xi32>
  hlfir.assign %2 to %0#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
  return
}
// CHECK-LABEL:   func.func @eoshift2(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: i32 {fir.bindc_name = "sh"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca i32
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK:           fir.store %[[VAL_1]] to %[[VAL_7]] : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_5]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_14]], %[[VAL_15]], %[[VAL_16]], %[[BOUNDARY]], %[[VAL_17]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// 2d boxed array shift by boxed array
func.func @eoshift3(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "sh"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %1:2 = hlfir.declare %arg1 {uniq_name = "sh"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %2 = hlfir.eoshift %0#0 %1#0 : (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?xi32>>) -> !hlfir.expr<?x?xi32>
  hlfir.assign %2 to %0#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
  return
}
// CHECK-LABEL:   func.func @eoshift3(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "sh"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "sh"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK:           %[[VAL_9:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_10:.*]] = fir.shape %[[VAL_5]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_9]](%[[VAL_10]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           fir.store %[[VAL_11]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_13:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_14:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]], %[[BOUNDARY]], %[[VAL_16]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// 2d boxed array shift by array expr
func.func @eoshift4(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"}, %arg1: !hlfir.expr<?xi32> {fir.bindc_name = "sh"}) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  %2 = hlfir.eoshift %0#0 %arg1 : (!fir.box<!fir.array<?x?xi32>>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
  hlfir.assign %2 to %0#0 : !hlfir.expr<?x?xi32>, !fir.box<!fir.array<?x?xi32>>
  return
}
// CHECK-LABEL:   func.func @eoshift4(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !hlfir.expr<?xi32> {fir.bindc_name = "sh"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "a"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?xi32>) -> !fir.shape<1>
// CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_1]](%[[VAL_8]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
// CHECK:           %[[VAL_10:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_11:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_12:.*]] = fir.shape %[[VAL_5]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_11]](%[[VAL_12]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           fir.store %[[VAL_13]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_9]]#1(%[[VAL_14]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_7]]#1 : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[BOUNDARY]], %[[VAL_20]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// 2d array expr shift by array expr
func.func @eoshift5(%arg0: !hlfir.expr<?x?xi32> {fir.bindc_name = "a"}, %arg1: !hlfir.expr<?xi32> {fir.bindc_name = "sh"}) {
  %2 = hlfir.eoshift %arg0 %arg1 : (!hlfir.expr<?x?xi32>, !hlfir.expr<?xi32>) -> !hlfir.expr<?x?xi32>
  hlfir.destroy %2 : !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL:   func.func @eoshift5(
// CHECK-SAME:                       %[[VAL_0:.*]]: !hlfir.expr<?x?xi32> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !hlfir.expr<?xi32> {fir.bindc_name = "sh"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           %[[VAL_7:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
// CHECK:           %[[VAL_8:.*]]:3 = hlfir.associate %[[VAL_0]](%[[VAL_7]]) {adapt.valuebyref} : (!hlfir.expr<?x?xi32>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>, i1)
// CHECK:           %[[VAL_9:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_10:.*]] = hlfir.get_extent %[[VAL_7]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_14:.*]] = fir.shape %[[VAL_9]], %[[VAL_10]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_8]]#1(%[[VAL_14]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_11:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?xi32>) -> !fir.shape<1>
// CHECK:           %[[VAL_12:.*]]:3 = hlfir.associate %[[VAL_1]](%[[VAL_11]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
// CHECK:           %[[VAL_13:.*]] = hlfir.get_extent %[[VAL_11]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_16:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_17:.*]] = fir.shape %[[VAL_5]], %[[VAL_5]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_18:.*]] = fir.embox %[[VAL_16]](%[[VAL_17]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           fir.store %[[VAL_18]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_13]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_20:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_19]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_23:.*]] = fir.convert %[[VAL_15]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_20]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_4]] : (index) -> i32
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[BOUNDARY]], %[[VAL_25]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// 2d array expr shift by array expr with explicit dim
func.func @eoshift6(%arg0: !hlfir.expr<?x?xi32> {fir.bindc_name = "a"}, %arg1: !hlfir.expr<?xi32> {fir.bindc_name = "sh"}, %dim : i16) {
  %2 = hlfir.eoshift %arg0 %arg1 dim %dim : (!hlfir.expr<?x?xi32>, !hlfir.expr<?xi32>, i16) -> !hlfir.expr<?x?xi32>
  hlfir.destroy %2 : !hlfir.expr<?x?xi32>
  return
}
// CHECK-LABEL:   func.func @eoshift6(
// CHECK-SAME:                       %[[VAL_0:.*]]: !hlfir.expr<?x?xi32> {fir.bindc_name = "a"},
// CHECK-SAME:                       %[[VAL_1:.*]]: !hlfir.expr<?xi32> {fir.bindc_name = "sh"},
// CHECK-SAME:                       %[[VAL_2:.*]]: i16) {
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           %[[VAL_8:.*]] = hlfir.shape_of %[[VAL_0]] : (!hlfir.expr<?x?xi32>) -> !fir.shape<2>
// CHECK:           %[[VAL_9:.*]]:3 = hlfir.associate %[[VAL_0]](%[[VAL_8]]) {adapt.valuebyref} : (!hlfir.expr<?x?xi32>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.ref<!fir.array<?x?xi32>>, i1)
// CHECK:           %[[VAL_10:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 0 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_11:.*]] = hlfir.get_extent %[[VAL_8]] {dim = 1 : index} : (!fir.shape<2>) -> index
// CHECK:           %[[VAL_16:.*]] = fir.shape %[[VAL_10]], %[[VAL_11]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_17:.*]] = fir.embox %[[VAL_9]]#1(%[[VAL_16]]) : (!fir.ref<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_12:.*]] = hlfir.shape_of %[[VAL_1]] : (!hlfir.expr<?xi32>) -> !fir.shape<1>
// CHECK:           %[[VAL_13:.*]]:3 = hlfir.associate %[[VAL_1]](%[[VAL_12]]) {adapt.valuebyref} : (!hlfir.expr<?xi32>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, i1)
// CHECK:           %[[VAL_14:.*]] = hlfir.get_extent %[[VAL_12]] {dim = 0 : index} : (!fir.shape<1>) -> index
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_2]] : (i16) -> i32
// CHECK:           %[[VAL_18:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xi32>>
// CHECK:           %[[VAL_19:.*]] = fir.shape %[[VAL_6]], %[[VAL_6]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_20:.*]] = fir.embox %[[VAL_18]](%[[VAL_19]]) : (!fir.heap<!fir.array<?x?xi32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xi32>>>
// CHECK:           fir.store %[[VAL_20]] to %[[VAL_7]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_21:.*]] = fir.shape %[[VAL_14]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_22:.*]] = fir.embox %[[VAL_13]]#1(%[[VAL_21]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_24:.*]] = fir.convert %[[VAL_7]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_25:.*]] = fir.convert %[[VAL_17]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<none>
// CHECK:           %[[VAL_26:.*]] = fir.convert %[[VAL_22]] : (!fir.box<!fir.array<?xi32>>) -> !fir.box<none>
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_24]], %[[VAL_25]], %[[VAL_26]], %[[BOUNDARY]], %[[VAL_15]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// shift of polymorphic array
func.func @eoshift7(%arg0: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, %arg1: !fir.ref<f32>) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "a"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "sh"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
  %c2_i32 = arith.constant 2 : i32
  %3 = fir.load %1#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  %4 = hlfir.eoshift %3 %c2_i32 : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>, i32) -> !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  hlfir.assign %4 to %1#0 realloc : !hlfir.expr<?x!fir.type<_QMtypesTt>?>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
  hlfir.destroy %4 : !hlfir.expr<?x!fir.type<_QMtypesTt>?>
  return
}
// CHECK-LABEL:   func.func @eoshift7(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: !fir.ref<f32>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca i32
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %[[VAL_8]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "a"} : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.dscope) -> (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>, !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>)
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %[[VAL_8]] {uniq_name = "sh"} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_9]]#0 : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) source_box %[[VAL_11]] : (!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>, !fir.shape<1>, !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_6]] : !fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>
// CHECK:           %[[BOUNDARY:.*]] = fir.absent !fir.box<none>
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_11]] : (!fir.class<!fir.heap<!fir.array<?x!fir.type<_QMtypesTt>>>>) -> !fir.box<none>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (i32) -> i64
// CHECK:           fir.call @_FortranAEoshiftVector(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[BOUNDARY]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, i64, !fir.box<none>, !fir.ref<i8>, i32) -> ()

// shift with the present scalar boundary and dim
func.func @_QPeoshift8(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"}) {
  %cst = arith.constant 3.000000e+00 : f32
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift8Earray"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %2 = hlfir.eoshift %1#0 %c2_i32 boundary %cst dim %c2_i32 : (!fir.box<!fir.array<?x?xf32>>, i32, f32, i32) -> !hlfir.expr<?x?xf32>
  hlfir.assign %2 to %1#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  hlfir.destroy %2 : !hlfir.expr<?x?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift8(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           %[[VAL_7:.*]] = fir.alloca f32
// CHECK:           %[[VAL_8:.*]] = fir.alloca i32
// CHECK:           %[[VAL_9:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_9]] {uniq_name = "_QFeoshift8Earray"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_8]] : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_5]] to %[[VAL_7]] : !fir.ref<f32>
// CHECK:           %[[VAL_11:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<f32>) -> !fir.box<f32>
// CHECK:           %[[VAL_12:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_13:.*]] = fir.shape %[[VAL_3]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_14:.*]] = fir.embox %[[VAL_12]](%[[VAL_13]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_6]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
// CHECK:           %[[VAL_15:.*]] = fir.embox %[[VAL_8]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_6]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_15]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_20:.*]] = fir.convert %[[VAL_11]] : (!fir.box<f32>) -> !fir.box<none>
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %[[VAL_20]], %[[VAL_4]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()

// shift with the present array boundary
func.func @_QPeoshift9(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"}, %arg1: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "boundary"}) {
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFeoshift9Earray"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
  %2:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QFeoshift9Eboundary"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %3 = hlfir.eoshift %1#0 %c2_i32 boundary %2#0 : (!fir.box<!fir.array<?x?xf32>>, i32, !fir.box<!fir.array<?xf32>>) -> !hlfir.expr<?x?xf32>
  hlfir.assign %3 to %1#0 : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
  hlfir.destroy %3 : !hlfir.expr<?x?xf32>
  return
}
// CHECK-LABEL:   func.func @_QPeoshift9(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "array"},
// CHECK-SAME:      %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "boundary"}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           %[[VAL_6:.*]] = fir.alloca i32
// CHECK:           %[[VAL_7:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift9Earray"} : (!fir.box<!fir.array<?x?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.box<!fir.array<?x?xf32>>)
// CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %[[VAL_7]] {uniq_name = "_QFeoshift9Eboundary"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_6]] : !fir.ref<i32>
// CHECK:           %[[VAL_10:.*]] = fir.zero_bits !fir.heap<!fir.array<?x?xf32>>
// CHECK:           %[[VAL_11:.*]] = fir.shape %[[VAL_3]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_12:.*]] = fir.embox %[[VAL_10]](%[[VAL_11]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
// CHECK:           fir.store %[[VAL_12]] to %[[VAL_5]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
// CHECK:           %[[VAL_13:.*]] = fir.embox %[[VAL_6]] : (!fir.ref<i32>) -> !fir.box<i32>
// CHECK:           %[[VAL_15:.*]] = fir.convert %[[VAL_5]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>) -> !fir.ref<!fir.box<none>>
// CHECK:           %[[VAL_16:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_17:.*]] = fir.convert %[[VAL_13]] : (!fir.box<i32>) -> !fir.box<none>
// CHECK:           %[[VAL_18:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.box<!fir.array<?xf32>>) -> !fir.box<none>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_2]] : (index) -> i32
// CHECK:           fir.call @_FortranAEoshift(%[[VAL_15]], %[[VAL_16]], %[[VAL_17]], %[[VAL_18]], %[[VAL_19]], %{{.*}}, %{{.*}}) : (!fir.ref<!fir.box<none>>, !fir.box<none>, !fir.box<none>, !fir.box<none>, i32, !fir.ref<i8>, i32) -> ()
