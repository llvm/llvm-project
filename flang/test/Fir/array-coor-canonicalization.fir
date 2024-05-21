// RUN: fir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL:   func.func @_QPtest1(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<120x2xi32>> {fir.bindc_name = "u"}) {
// CHECK:           %[[VAL_6:.*]] = fir.shape
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_0]](%[[VAL_6]])
// CHECK:           %[[VAL_8:.*]] = fir.slice
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_11:.*]] = fir.array_coor %[[VAL_7]](%[[VAL_6]]) {{\[}}%[[VAL_8]]]
func.func @_QPtest1(%arg0: !fir.ref<!fir.array<120x2xi32>> {fir.bindc_name = "u"}) {
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c2 = arith.constant 2 : index
  %c120 = arith.constant 120 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c120, %c2 : (index, index) -> !fir.shape<2>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest1Eu"} : (!fir.ref<!fir.array<120x2xi32>>, !fir.shape<2>, !fir.dscope) -> !fir.ref<!fir.array<120x2xi32>>
  %3 = fir.slice %c1, %c120, %c1, %c1, %c2, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %4 = fir.embox %2(%1) [%3] : (!fir.ref<!fir.array<120x2xi32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<120x2xi32>>
  fir.do_loop %arg1 = %c1 to %c2 step %c1 unordered {
    fir.do_loop %arg2 = %c1 to %c120 step %c1 unordered {
      %5 = fir.array_coor %4 %arg2, %arg1 : (!fir.box<!fir.array<120x2xi32>>, index, index) -> !fir.ref<i32>
      fir.store %c2_i32 to %5 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "u"}) {
// CHECK:           %[[VAL_8:.*]] = fir.shift
// CHECK:           %[[VAL_9:.*]] = fir.declare %[[VAL_0]](%[[VAL_8]])
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_17:.*]] = fir.array_coor %[[VAL_9]](%[[VAL_8]])
func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "u"}) {
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c11 = arith.constant 11 : index
  %c10 = arith.constant 10 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shift %c10, %c11 : (index, index) -> !fir.shift<2>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest2Eu"} : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %3 = fir.rebox %2(%1) : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xi32>>
  %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  %5:3 = fir.box_dims %3, %c1 : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
  fir.do_loop %arg1 = %c1 to %5#1 step %c1 unordered {
    fir.do_loop %arg2 = %c1 to %4#1 step %c1 unordered {
      %6 = arith.addi %arg2, %c9 : index
      %7 = arith.addi %arg1, %c10 : index
      %8 = fir.array_coor %3(%1) %6, %7 : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, index, index) -> !fir.ref<i32>
      fir.store %c2_i32 to %8 : !fir.ref<i32>
    }
  }
  return
}

// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "u"}) {
// CHECK:           %[[VAL_10:.*]] = fir.shift
// CHECK:           %[[VAL_11:.*]] = fir.declare %[[VAL_0]](%[[VAL_10]])
// CHECK:           %[[VAL_12:.*]] = fir.slice
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_15:.*]] = fir.array_coor %[[VAL_11]](%[[VAL_10]]) {{\[}}%[[VAL_12]]]
func.func @_QPtest3(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "u"}) {
  %c2 = arith.constant 2 : index
  %c12 = arith.constant 12 : index
  %c11 = arith.constant 11 : index
  %c111 = arith.constant 111 : index
  %c1 = arith.constant 1 : index
  %c120 = arith.constant 120 : index
  %c10 = arith.constant 10 : index
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shift %c10, %c11 : (index, index) -> !fir.shift<2>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFtest3Eu"} : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xi32>>
  %3 = fir.rebox %2(%1) : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xi32>>
  %4 = fir.slice %c10, %c120, %c1, %c11, %c12, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %5 = fir.rebox %3(%1) [%4] : (!fir.box<!fir.array<?x?xi32>>, !fir.shift<2>, !fir.slice<2>) -> !fir.box<!fir.array<111x2xi32>>
  fir.do_loop %arg1 = %c1 to %c2 step %c1 unordered {
    fir.do_loop %arg2 = %c1 to %c111 step %c1 unordered {
      %6 = fir.array_coor %5 %arg2, %arg1 : (!fir.box<!fir.array<111x2xi32>>, index, index) -> !fir.ref<i32>
      fir.store %c2_i32 to %6 : !fir.ref<i32>
    }
  }
  return
}

// TODO: fir.array_coor with slices changing the number of dimensions
// is not supported yet.
// CHECK-LABEL:   func.func @_QPtest4() {
// CHECK:           %[[VAL_3:.*]] = fir.alloca !fir.array<100x100x100xi32> {bindc_name = "u", uniq_name = "_QFtest4Eu"}
// CHECK:           %[[VAL_4:.*]] = fir.shape
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_3]](%[[VAL_4]]) {uniq_name = "_QFtest4Eu"} : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<100x100x100xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.slice
// CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_5]](%[[VAL_4]]) {{\[}}%[[VAL_7]]] : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>, !fir.slice<3>) -> !fir.box<!fir.array<100x100xi32>>
// CHECK:           fir.do_loop
// CHECK:             fir.do_loop
// CHECK:               %[[VAL_11:.*]] = fir.array_coor %[[VAL_8]]
func.func @_QPtest4() {
  %c1 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c100 = arith.constant 100 : index
  %0 = fir.alloca !fir.array<100x100x100xi32> {bindc_name = "u", uniq_name = "_QFtest4Eu"}
  %1 = fir.shape %c100, %c100, %c100 : (index, index, index) -> !fir.shape<3>
  %2 = fir.declare %0(%1) {uniq_name = "_QFtest4Eu"} : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<100x100x100xi32>>
  %3 = fir.undefined index
  %4 = fir.slice %c1, %c100, %c1, %c1, %3, %3, %c1, %c100, %c1 : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
  %5 = fir.embox %2(%1) [%4] : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>, !fir.slice<3>) -> !fir.box<!fir.array<100x100xi32>>
  fir.do_loop %arg0 = %c1 to %c100 step %c1 unordered {
    fir.do_loop %arg1 = %c1 to %c100 step %c1 unordered {
      %6 = fir.array_coor %5 %arg1, %arg0 : (!fir.box<!fir.array<100x100xi32>>, index, index) -> !fir.ref<i32>
      fir.store %c2_i32 to %6 : !fir.ref<i32>
    }
  }
  return
}
