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

// Rank-reducing slice: the middle dimension is scalar-selected by the slice
// (its upper bound is fir.undefined). The canonicalization pulls in the embox
// and rewrites the rank-2 array_coor into a rank-3 array_coor on the original
// 3D reference, with the scalar dim's index synthesized from the slice's
// lower bound.
// CHECK-LABEL:   func.func @_QPtest4() {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_4:.*]] = fir.alloca !fir.array<100x100x100xi32> {bindc_name = "u", uniq_name = "_QFtest4Eu"}
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_3]], %[[VAL_3]], %[[VAL_3]] : (index, index, index) -> !fir.shape<3>
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_4]](%[[VAL_5]]) {uniq_name = "_QFtest4Eu"} : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>) -> !fir.ref<!fir.array<100x100x100xi32>>
// CHECK:           %[[VAL_8:.*]] = fir.slice
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_3]]
// CHECK:             fir.do_loop %[[VAL_10:.*]] = %[[VAL_1]] to %[[VAL_3]]
// CHECK:               %[[VAL_11:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_5]]) {{\[}}%[[VAL_8]]] %[[VAL_10]], %[[VAL_1]], %[[VAL_9]] : (!fir.ref<!fir.array<100x100x100xi32>>, !fir.shape<3>, !fir.slice<3>, index, index, index) -> !fir.ref<i32>
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

// CHECK-LABEL:   func.func @test5_1(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_3]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test5_1(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %sh = fir.shift %c10 : (index) -> !fir.shift<1>
  %2 = fir.rebox %arg0(%sh) : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xi32>>
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test5_2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_5:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) {{\[}}%[[VAL_4]]] %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_5]] : !fir.ref<i32>
// CHECK:         }
func.func @test5_2(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %s = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test5_3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_5:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) {{\[}}%[[VAL_4]]] %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_5]] : !fir.ref<i32>
// CHECK:         }
func.func @test5_3(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c1, %c10, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test6_1(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = fir.shift %[[VAL_1]] : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_4:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_5:.*]] = fir.rebox %[[VAL_0]](%[[VAL_3]]) {{\[}}%[[VAL_4]]] : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_5]] {{\[}}%[[VAL_4]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test6_1(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %sh = fir.shift %c1 : (index) -> !fir.shift<1>
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0(%sh) [%s] : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test6_2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]] {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_7]] : !fir.ref<i32>
// CHECK:         }
func.func @test6_2(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test6_3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]] {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_7]] : !fir.ref<i32>
// CHECK:         }
func.func @test6_3(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test7_1(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.array_coor %[[VAL_0]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_2]] : !fir.ref<i32>
// CHECK:         }
func.func @test7_1(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %sh = fir.shift %c10 : (index) -> !fir.shift<1>
  %2 = fir.rebox %arg0(%sh) : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test7_2(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test7_2(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test7_3(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test7_3(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test8_1(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: index,
// CHECK-SAME:                       %[[VAL_2:.*]]: index) -> !fir.ref<i32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_6:.*]] = fir.shift %[[VAL_1]] : (index) -> !fir.shift<1>
// CHECK:           %[[VAL_7:.*]] = fir.slice %[[VAL_4]], %[[VAL_5]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_2]], %[[VAL_1]] overflow<nsw> : index
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_3]] overflow<nsw> : index
// CHECK:           %[[VAL_10:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_6]]) {{\[}}%[[VAL_7]]] %[[VAL_9]] : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_10]] : !fir.ref<i32>
// CHECK:         }
func.func @test8_1(%arg0: !fir.box<!fir.array<?xi32>>, %lb: index, %idx: index) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %sh = fir.shift %lb : (index) -> !fir.shift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0(%sh) [%s] : (!fir.box<!fir.array<?xi32>>, !fir.shift<1>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 %idx : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test8_2(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_1]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test8_2(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test8_3(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>,
// CHECK-SAME:                       %[[VAL_1:.*]]: index,
// CHECK-SAME:                       %[[VAL_2:.*]]: index) -> !fir.ref<i32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_6:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_4]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_7:.*]] = fir.slice %[[VAL_4]], %[[VAL_5]], %[[VAL_3]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_2]], %[[VAL_1]] overflow<nsw> : index
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_3]] overflow<nsw> : index
// CHECK:           %[[VAL_10:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_6]]) {{\[}}%[[VAL_7]]] %[[VAL_9]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_10]] : !fir.ref<i32>
// CHECK:         }
func.func @test8_3(%arg0: !fir.ref<!fir.array<10xi32>>, %lb: index, %idx: index) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %lb, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2 %idx : (!fir.box<!fir.array<10xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test9(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_3]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test9(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test10(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_3]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test10(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0[%s] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 [%s] %c1 : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test11(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = fir.array_coor %[[VAL_0]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_2]] : !fir.ref<i32>
// CHECK:         }
func.func @test11(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %2 = fir.rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test12(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_3:.*]] = fir.slice %[[VAL_1]], %[[VAL_2]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]] {{\[}}%[[VAL_3]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_4]] : !fir.ref<i32>
// CHECK:         }
func.func @test12(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0[%s] : (!fir.box<!fir.array<?xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2 %c1 : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test13(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test13(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test14(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test14(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test15(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_1]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_3:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_2]]) %[[VAL_1]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_3]] : !fir.ref<i32>
// CHECK:         }
func.func @test15(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c10 = arith.constant 10 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %2 = fir.embox %arg0(%1) : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) %c10 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test16(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_2]] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test16(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0(%1) [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) %c10 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test17(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 19 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_6]] : !fir.ref<i32>
// CHECK:         }
func.func @test17(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2(%1) [%s] %c1 : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test18(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_1]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_0]] {{\[}}%[[VAL_5]]] : (!fir.ref<!fir.array<10xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_4]]) {{\[}}%[[VAL_5]]] %[[VAL_1]] : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_7]] : !fir.ref<i32>
// CHECK:         }
func.func @test18(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %1 = fir.shape_shift %c1, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0 [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) [%s] %c1 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, !fir.slice<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test19(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape_shift %[[VAL_1]], %[[VAL_1]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_3:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_2]]) %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_3]] : !fir.ref<i32>
// CHECK:         }
func.func @test19(%arg0: !fir.box<!fir.array<?xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c19 = arith.constant 19 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c10, %c19, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.rebox %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>>
  %3 = fir.array_coor %2(%1) %c10 : (!fir.box<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// CHECK-LABEL:   func.func @test20(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape_shift %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shapeshift<1>
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_1]], %[[VAL_3]], %[[VAL_1]] : (index, index, index) -> !fir.slice<1>
// CHECK:           %[[VAL_6:.*]] = fir.embox %[[VAL_0]] {{\[}}%[[VAL_5]]] : (!fir.ref<!fir.array<10xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_6]](%[[VAL_4]]) %[[VAL_2]] : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
// CHECK:           return %[[VAL_7]] : !fir.ref<i32>
// CHECK:         }
func.func @test20(%arg0: !fir.ref<!fir.array<10xi32>>) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %1 = fir.shape_shift %c10, %c10 : (index, index) -> !fir.shapeshift<1>
  %s = fir.slice %c1, %c9, %c1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.embox %arg0 [%s] : (!fir.ref<!fir.array<10xi32>>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
  %3 = fir.array_coor %2(%1) %c10 : (!fir.box<!fir.array<10xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
  return %3 : !fir.ref<i32>
}

// Rank-reducing slice reached through a fir.box_addr indirection.
// Mirrors the pattern produced by HLFIR-to-FIR lowering for a 2D contiguous
// section assignment: `cli(:m, i) = ...`. The embox creates a 1D box of a
// 2D ref by selecting one element in the second dim; box_addr yields a 1D
// ref; the array_coor accesses elements with one index. The canonicalization
// must look through the box_addr to find the embox and rewrite the access
// as a 2D array_coor on the original ref so later lowering can see the
// full 2D address pattern.
// CHECK-LABEL:   func.func @test21_box_addr_rank_reducing(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<16x6xf32>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                      %[[VAL_2:.*]]: i64) -> !fir.ref<f32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant -2 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_7:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_5]], %[[VAL_4]], %[[VAL_6]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:           %[[VAL_8:.*]] = fir.undefined index
// CHECK:           %[[VAL_9:.*]] = fir.slice %[[VAL_3]], %[[VAL_1]], %[[VAL_3]], %[[VAL_2]], %[[VAL_8]], %[[VAL_8]] : (index, index, index, i64, index, index) -> !fir.slice<2>
// CHECK:           %[[VAL_10:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_7]]) {{\[}}%[[VAL_9]]] %[[VAL_3]], %[[VAL_2]] : (!fir.ref<!fir.array<16x6xf32>>, !fir.shapeshift<2>, !fir.slice<2>, index, i64) -> !fir.ref<f32>
// CHECK:           return %[[VAL_10]] : !fir.ref<f32>
// CHECK:         }
func.func @test21_box_addr_rank_reducing(%arg0: !fir.ref<!fir.array<16x6xf32>>, %m: index, %i: i64) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c_neg2 = arith.constant -2 : index
  %c16 = arith.constant 16 : index
  %c6 = arith.constant 6 : index
  %ss = fir.shape_shift %c1, %c16, %c_neg2, %c6 : (index, index, index, index) -> !fir.shapeshift<2>
  %u = fir.undefined index
  %s = fir.slice %c1, %m, %c1, %i, %u, %u : (index, index, index, i64, index, index) -> !fir.slice<2>
  %b = fir.embox %arg0(%ss) [%s] : (!fir.ref<!fir.array<16x6xf32>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
  %ba = fir.box_addr %b : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
  %sh1 = fir.shape %m : (index) -> !fir.shape<1>
  %ac = fir.array_coor %ba(%sh1) %c1 : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Rank-reducing slice via fir.rebox with no shift: 3D source box -> 2D box.
// array_coor on the sliced box is folded to a full-rank array_coor on the
// source box; fir.shape is synthesized from fir.box_dims when rebox has no
// shape operand (fir.shift is not used: rebox base_addr is pre-adjusted).
// CHECK-LABEL:   func.func @test22_rebox_rank_reducing({{.*}}) -> !fir.ref<f32> {
// CHECK:           %[[T22_C2:.*]] = arith.constant 2 : index
// CHECK:           %[[T22_C0:.*]] = arith.constant 0 : index
// CHECK:           %[[T22_C1:.*]] = arith.constant 1 : index
// CHECK:           %[[T22_C100:.*]] = arith.constant 100 : index
// CHECK:           %[[T22_U:.*]] = fir.undefined index
// CHECK:           %[[T22_SL:.*]] = fir.slice %[[T22_C1]], %[[T22_C100]], %[[T22_C1]], %{{.*}}, %[[T22_U]], %[[T22_U]], %[[T22_C1]], %[[T22_C100]], %[[T22_C1]] : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
// CHECK:           %[[T22_BD0:.*]]:3 = fir.box_dims %{{.*}}, %[[T22_C0]]
// CHECK:           %[[T22_BD1:.*]]:3 = fir.box_dims %{{.*}}, %[[T22_C1]]
// CHECK:           %[[T22_BD2:.*]]:3 = fir.box_dims %{{.*}}, %[[T22_C2]]
// CHECK:           %[[T22_SH:.*]] = fir.shape %[[T22_BD0]]#1, %[[T22_BD1]]#1, %[[T22_BD2]]#1 : (index, index, index) -> !fir.shape<3>
// CHECK:           %[[T22_AC:.*]] = fir.array_coor %{{.*}}(%[[T22_SH]]) {{\[}}%[[T22_SL]]] %[[T22_C1]], %{{.*}}, %[[T22_C1]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shape<3>, !fir.slice<3>, index, index, index) -> !fir.ref<f32>
// CHECK:           return %[[T22_AC]] : !fir.ref<f32>
// CHECK:         }
func.func @test22_rebox_rank_reducing(%arg0: !fir.box<!fir.array<?x?x?xf32>>, %j: index) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %u = fir.undefined index
  %s = fir.slice %c1, %c100, %c1, %j, %u, %u, %c1, %c100, %c1 : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
  %r = fir.rebox %arg0 [%s] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xf32>>
  %ac = fir.array_coor %r %c1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Rank-reducing slice via fir.rebox with an explicit shift. The fir.rebox
// verifier requires the shift's rank to match the *source* box rank when a
// slice is present, so the shift here is rank-3 (same as the new array_coor
// that we want to emit on the source box). The canonicalization can fold
// through to a 3D array_coor on the source box, synthesizing the dropped
// middle dim's index from the slice's lower bound.
// CHECK-LABEL:   func.func @test23_rebox_rank_reducing_with_shift(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?x?x?xf32>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index) -> !fir.ref<f32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_4:.*]] = fir.undefined index
// CHECK:           %[[VAL_5:.*]] = fir.slice %[[VAL_2]], %[[VAL_3]], %[[VAL_2]], %[[VAL_1]], %[[VAL_4]], %[[VAL_4]], %[[VAL_2]], %[[VAL_3]], %[[VAL_2]] : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
// CHECK:           %[[VAL_6:.*]] = fir.shift %[[VAL_2]], %[[VAL_2]], %[[VAL_2]] : (index, index, index) -> !fir.shift<3>
// CHECK:           %[[VAL_7:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_6]]) {{\[}}%[[VAL_5]]] %[[VAL_2]], %[[VAL_1]], %[[VAL_2]] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, !fir.slice<3>, index, index, index) -> !fir.ref<f32>
// CHECK:           return %[[VAL_7]] : !fir.ref<f32>
// CHECK:         }
func.func @test23_rebox_rank_reducing_with_shift(%arg0: !fir.box<!fir.array<?x?x?xf32>>, %j: index) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c100 = arith.constant 100 : index
  %u = fir.undefined index
  %s = fir.slice %c1, %c100, %c1, %j, %u, %u, %c1, %c100, %c1 : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
  %sh = fir.shift %c1, %c1, %c1 : (index, index, index) -> !fir.shift<3>
  %r = fir.rebox %arg0(%sh) [%s] : (!fir.box<!fir.array<?x?x?xf32>>, !fir.shift<3>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xf32>>
  %ac = fir.array_coor %r %c1, %c1 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Rank-reducing canonicalization with non-default lower bounds in non-scalar
// dimensions keeps indices in shape-shift coordinate space so CodeGen can
// consume them without additional special cases.
// CHECK-LABEL:   func.func @test24_rank_reducing_shifted_nonscalar(
// CHECK-SAME:      %[[VAL_0:.*]]: !fir.ref<!fir.array<20x20x20xf32>>,
// CHECK-SAME:      %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index, %[[VAL_5:.*]]: index) -> !fir.ref<f32> {
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 20 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_9:.*]] = arith.constant -2 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 5 : index
// CHECK:           %[[VAL_11:.*]] = fir.shape_shift %[[VAL_8]], %[[VAL_7]], %[[VAL_9]], %[[VAL_7]], %[[VAL_10]], %[[VAL_7]] : (index, index, index, index, index, index) -> !fir.shapeshift<3>
// CHECK:           %[[VAL_12:.*]] = fir.undefined index
// CHECK:           %[[VAL_13:.*]] = fir.slice %[[VAL_6]], %[[VAL_1]], %[[VAL_6]], %[[VAL_2]], %[[VAL_12]], %[[VAL_12]], %[[VAL_6]], %[[VAL_3]], %[[VAL_6]] : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_4]], %c9 : index
// CHECK:           %[[VAL_15:.*]] = arith.addi %[[VAL_5]], %c4 : index
// CHECK:           %[[VAL_16:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_11]]) {{\[}}%[[VAL_13]]] %[[VAL_14]], %[[VAL_2]], %[[VAL_15]] : (!fir.ref<!fir.array<20x20x20xf32>>, !fir.shapeshift<3>, !fir.slice<3>, index, index, index) -> !fir.ref<f32>
// CHECK:           return %[[VAL_16]] : !fir.ref<f32>
// CHECK:         }
func.func @test24_rank_reducing_shifted_nonscalar(
    %arg0: !fir.ref<!fir.array<20x20x20xf32>>, %m: index, %j: index, %n: index,
    %i: index, %k: index) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c20 = arith.constant 20 : index
  %c10 = arith.constant 10 : index
  %c_neg2 = arith.constant -2 : index
  %c5 = arith.constant 5 : index
  %ss = fir.shape_shift %c10, %c20, %c_neg2, %c20, %c5, %c20 : (index, index, index, index, index, index) -> !fir.shapeshift<3>
  %u = fir.undefined index
  %s = fir.slice %c1, %m, %c1, %j, %u, %u, %c1, %n, %c1 : (index, index, index, index, index, index, index, index, index) -> !fir.slice<3>
  %b = fir.embox %arg0(%ss) [%s] : (!fir.ref<!fir.array<20x20x20xf32>>, !fir.shapeshift<3>, !fir.slice<3>) -> !fir.box<!fir.array<?x?xf32>>
  %ac = fir.array_coor %b %i, %k : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Deferred-length character array accessed through box_addr: typeparams on the
// ref array_coor carry the element LEN, but after pulling the rebox through
// box_addr the memref becomes a box and must not keep those typeparams.
// CHECK-LABEL:   func.func @test25_box_addr_char_typeparams(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.char<1,?>>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index) -> !fir.ref<!fir.char<1,?>> {
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_2]]) %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:           return %[[VAL_3]] : !fir.ref<!fir.char<1,?>>
// CHECK:         }
func.func @test25_box_addr_char_typeparams(%arg0: !fir.box<!fir.array<?x!fir.char<1,?>>>, %idx: index) -> !fir.ref<!fir.char<1,?>> {
  %c0 = arith.constant 0 : index
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  %elemsize = fir.box_elesize %rebox : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
  %ref = fir.box_addr %rebox : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %shape = fir.shape %idx : (index) -> !fir.shape<1>
  %co = fir.array_coor %ref(%shape) %idx typeparams %elemsize : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  return %co : !fir.ref<!fir.char<1,?>>
}

// Rank-reducing deferred-length character slice through fir.box_addr: the ref
// array_coor carries element LEN in typeparams; canonicalization must fold to a
// full-rank array_coor on the original ref and keep those typeparams.
// CHECK-LABEL:   func.func @test26_box_addr_char_rank_reducing_typeparams(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<16x6x!fir.char<1,?>>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                      %[[VAL_2:.*]]: i64,
// CHECK-SAME:                      %[[VAL_3:.*]]: index) -> !fir.ref<!fir.char<1,?>> {
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant -2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_6]], %[[VAL_5]], %[[VAL_7]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:           %[[VAL_9:.*]] = fir.undefined index
// CHECK:           %[[VAL_10:.*]] = fir.slice %[[VAL_4]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]], %[[VAL_9]], %[[VAL_9]] : (index, index, index, i64, index, index) -> !fir.slice<2>
// CHECK:           %[[VAL_11:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_10]]] %[[VAL_4]], %[[VAL_2]] typeparams %[[VAL_3]] : (!fir.ref<!fir.array<16x6x!fir.char<1,?>>>, !fir.shapeshift<2>, !fir.slice<2>, index, i64, index) -> !fir.ref<!fir.char<1,?>>
// CHECK:           return %[[VAL_11]] : !fir.ref<!fir.char<1,?>>
// CHECK:         }
func.func @test26_box_addr_char_rank_reducing_typeparams(
    %arg0: !fir.ref<!fir.array<16x6x!fir.char<1,?>>>, %m: index, %i: i64,
    %elemsize: index) -> !fir.ref<!fir.char<1,?>> {
  %c1 = arith.constant 1 : index
  %c_neg2 = arith.constant -2 : index
  %c16 = arith.constant 16 : index
  %c6 = arith.constant 6 : index
  %ss = fir.shape_shift %c1, %c16, %c_neg2, %c6 : (index, index, index, index) -> !fir.shapeshift<2>
  %u = fir.undefined index
  %s = fir.slice %c1, %m, %c1, %i, %u, %u : (index, index, index, i64, index, index) -> !fir.slice<2>
  %b = fir.embox %arg0(%ss) [%s] : (!fir.ref<!fir.array<16x6x!fir.char<1,?>>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x!fir.char<1,?>>>
  %ba = fir.box_addr %b : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> !fir.ref<!fir.array<?x!fir.char<1,?>>>
  %sh1 = fir.shape %m : (index) -> !fir.shape<1>
  %ac = fir.array_coor %ba(%sh1) %c1 typeparams %elemsize : (!fir.ref<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index, index) -> !fir.ref<!fir.char<1,?>>
  return %ac : !fir.ref<!fir.char<1,?>>
}

// Parametric derived type accessed through box_addr: typeparams on the ref
// array_coor carry the element LEN, but after pulling the rebox through box_addr
// the memref becomes a box and must not keep those typeparams.
// CHECK-LABEL:   func.func @test27_box_addr_derived_typeparams(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                      %[[VAL_2:.*]]: i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>> {
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) %[[VAL_1]] : (!fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>>
// CHECK:           return %[[VAL_4]] : !fir.ref<!fir.type<dt(l:i32){x:i32}>>
// CHECK:         }
func.func @test27_box_addr_derived_typeparams(
    %arg0: !fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>, %idx: index,
    %len: i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>> {
  %rebox = fir.rebox %arg0 : (!fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>) -> !fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>
  %ref = fir.box_addr %rebox : (!fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>) -> !fir.ref<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>
  %shape = fir.shape %idx : (index) -> !fir.shape<1>
  %co = fir.array_coor %ref(%shape) %idx typeparams %len : (!fir.ref<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>, !fir.shape<1>, index, i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>>
  return %co : !fir.ref<!fir.type<dt(l:i32){x:i32}>>
}

// Rank-reducing slice of an array of parametric derived types through
// fir.box_addr: typeparams on the ref array_coor must be kept after folding to
// a full-rank array_coor on the original ref.
// CHECK-LABEL:   func.func @test28_box_addr_derived_rank_reducing_typeparams(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<16x6x!fir.type<dt(l:i32){x:i32}>>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                      %[[VAL_2:.*]]: i64,
// CHECK-SAME:                      %[[VAL_3:.*]]: i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>> {
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant -2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 6 : index
// CHECK:           %[[VAL_8:.*]] = fir.shape_shift %[[VAL_4]], %[[VAL_6]], %[[VAL_5]], %[[VAL_7]] : (index, index, index, index) -> !fir.shapeshift<2>
// CHECK:           %[[VAL_9:.*]] = fir.undefined index
// CHECK:           %[[VAL_10:.*]] = fir.slice %[[VAL_4]], %[[VAL_1]], %[[VAL_4]], %[[VAL_2]], %[[VAL_9]], %[[VAL_9]] : (index, index, index, i64, index, index) -> !fir.slice<2>
// CHECK:           %[[VAL_11:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_8]]) {{\[}}%[[VAL_10]]] %[[VAL_4]], %[[VAL_2]] typeparams %[[VAL_3]] : (!fir.ref<!fir.array<16x6x!fir.type<dt(l:i32){x:i32}>>>, !fir.shapeshift<2>, !fir.slice<2>, index, i64, i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>>
// CHECK:           return %[[VAL_11]] : !fir.ref<!fir.type<dt(l:i32){x:i32}>>
// CHECK:         }
func.func @test28_box_addr_derived_rank_reducing_typeparams(
    %arg0: !fir.ref<!fir.array<16x6x!fir.type<dt(l:i32){x:i32}>>>, %m: index,
    %i: i64, %len: i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>> {
  %c1 = arith.constant 1 : index
  %c_neg2 = arith.constant -2 : index
  %c16 = arith.constant 16 : index
  %c6 = arith.constant 6 : index
  %ss = fir.shape_shift %c1, %c16, %c_neg2, %c6 : (index, index, index, index) -> !fir.shapeshift<2>
  %u = fir.undefined index
  %s = fir.slice %c1, %m, %c1, %i, %u, %u : (index, index, index, i64, index, index) -> !fir.slice<2>
  %b = fir.embox %arg0(%ss) [%s] : (!fir.ref<!fir.array<16x6x!fir.type<dt(l:i32){x:i32}>>>, !fir.shapeshift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>
  %ba = fir.box_addr %b : (!fir.box<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>) -> !fir.ref<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>
  %sh1 = fir.shape %m : (index) -> !fir.shape<1>
  %ac = fir.array_coor %ba(%sh1) %c1 typeparams %len : (!fir.ref<!fir.array<?x!fir.type<dt(l:i32){x:i32}>>>, !fir.shape<1>, index, i32) -> !fir.ref<!fir.type<dt(l:i32){x:i32}>>
  return %ac : !fir.ref<!fir.type<dt(l:i32){x:i32}>>
}

// Merged type mismatch on fir.box_addr (no fir.convert): box<2x2> -> ref<4x3>.
// Canonicalization must not peel box_addr and fold to array_coor on %arg0.
// CHECK-LABEL:   func.func @test29_merged_box_addr_no_fold(
// CHECK:           fir.embox
// CHECK:           fir.box_addr
// CHECK-NOT:       fir.array_coor %{{.*}}({{.*}}) [{{.*}}]
func.func @test29_merged_box_addr_no_fold(%arg0: !fir.ref<!fir.array<4x3xf32>>) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c3 = arith.constant 3 : index
  %sh = fir.shape %c4, %c3 : (index, index) -> !fir.shape<2>
  %sl = fir.slice %c1, %c2, %c1, %c1, %c2, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
  %b = fir.embox %arg0(%sh) [%sl] : (!fir.ref<!fir.array<4x3xf32>>, !fir.shape<2>, !fir.slice<2>) -> !fir.box<!fir.array<2x2xf32>>
  %ba = fir.box_addr %b : (!fir.box<!fir.array<2x2xf32>>) -> !fir.ref<!fir.array<4x3xf32>>
  %ac = fir.array_coor %ba(%sh) %c1, %c1 : (!fir.ref<!fir.array<4x3xf32>>, !fir.shape<2>, index, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Rank-reducing 2D->1D slice via fir.rebox: array_coor uses the sliced 1D box
// directly; fold to 2-index array_coor on the 2D source.
// CHECK-LABEL:   func.func @test30_rebox_rank_reducing_2d({{.*}}) -> !fir.ref<f32> {
// CHECK:           %[[T29_C0:.*]] = arith.constant 0 : index
// CHECK:           %[[T29_C1:.*]] = arith.constant 1 : index
// CHECK:           %[[T29_C2:.*]] = arith.constant 2 : index
// CHECK:           %[[T29_U:.*]] = fir.undefined index
// CHECK:           %[[T29_SL:.*]] = fir.slice %{{.*}}, %{{.*}}, %[[T29_C1]], %[[T29_C2]], %[[T29_U]], %[[T29_U]] : (index, index, index, index, index, index) -> !fir.slice<2>
// CHECK:           %[[T29_BD0:.*]]:3 = fir.box_dims %{{.*}}, %[[T29_C0]]
// CHECK:           %[[T29_BD1:.*]]:3 = fir.box_dims %{{.*}}, %[[T29_C1]]
// CHECK:           %[[T29_SH:.*]] = fir.shape %[[T29_BD0]]#1, %[[T29_BD1]]#1 : (index, index) -> !fir.shape<2>
// CHECK:           %[[T29_AC:.*]] = fir.array_coor %{{.*}}(%[[T29_SH]]) {{\[}}%[[T29_SL]]] %{{.*}}, %[[T29_C2]] : (!fir.box<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.slice<2>, index, index) -> !fir.ref<f32>
// CHECK:           return %[[T29_AC]] : !fir.ref<f32>
// CHECK:         }
func.func @test30_rebox_rank_reducing_2d(
    %arg0: !fir.box<!fir.array<?x?xf32>>, %si: index, %ei: index,
    %row: index) -> !fir.ref<f32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %u = fir.undefined index
  %s = fir.slice %si, %ei, %c1, %c2, %u, %u : (index, index, index, index, index, index) -> !fir.slice<2>
  %b = fir.rebox %arg0 [%s] : (!fir.box<!fir.array<?x?xf32>>, !fir.slice<2>) -> !fir.box<!fir.array<?xf32>>
  %ac = fir.array_coor %b %row : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
  return %ac : !fir.ref<f32>
}

// Pull a previous fir.array_coor through a fir.convert into the current
// fir.array_coor. This mirrors how ConvertHLFIRtoFIR lowers a contiguous
// slice designator (e.g. `dx(1:3)`): the outer array_coor produces the
// address of the first element, fir.convert reinterprets it as a fixed-size
// array ref, and the inner array_coor indexes into that view.
//
// 1-D case: dx(1:3)(k) -> dx(k). When outer's index is the constant 1, the
// arith identity c1 + (k - 1) == k makes the canonicalizer collapse the
// combined index back to the inner index.
// CHECK-LABEL:   func.func @test31_merge_array_coor_1d(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<?xf64>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index,
// CHECK-SAME:                      %[[VAL_2:.*]]: index) -> !fir.ref<f64> {
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_3]]) %[[VAL_2]] : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
// CHECK:           return %[[VAL_4]] : !fir.ref<f64>
// CHECK-NOT:       fir.convert
// CHECK-NOT:       fir.array_coor %{{.*}} -> !fir.ref<!fir.array
func.func @test31_merge_array_coor_1d(%base: !fir.ref<!fir.array<?xf64>>,
                                      %n: index,
                                      %k: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %base_shape = fir.shape %n : (index) -> !fir.shape<1>
  %inner_shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %outer = fir.array_coor %base(%base_shape) %c1 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3xf64>>
  %inner = fir.array_coor %view(%inner_shape) %k : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// 2-D base with 1-D inner view: dx(4:6, 4)(k) -> dx(4 + k - 1, 4). Outer
// index is not 1 here, so the index arithmetic doesn't fully collapse;
// canonicalizer reassociates to a single arith.addi.
// CHECK-LABEL:   func.func @test32a_merge_array_coor_2d_offset(
// CHECK-SAME:        %[[ARG0:[A-Za-z0-9_]+]]: !fir.ref<!fir.array<?x?xf64>>,
// CHECK-SAME:        %[[ARG1:[A-Za-z0-9_]+]]: index, %[[ARG2:[A-Za-z0-9_]+]]: index, %[[ARG3:[A-Za-z0-9_]+]]: index)
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[SH:.*]] = fir.shape %[[ARG1]], %[[ARG2]]
// CHECK:           %[[IDX:.*]] = arith.addi %[[ARG3]], %[[C3]]
// CHECK:           %[[AC:.*]] = fir.array_coor %[[ARG0]](%[[SH]]) %[[IDX]], %[[C4]]
// CHECK:           return %[[AC]] : !fir.ref<f64>
// CHECK-NOT:       fir.convert
func.func @test32a_merge_array_coor_2d_offset(%base: !fir.ref<!fir.array<?x?xf64>>,
                                              %d1: index, %d2: index,
                                              %k: index) -> !fir.ref<f64> {
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %base_shape = fir.shape %d1, %d2 : (index, index) -> !fir.shape<2>
  %inner_shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %outer = fir.array_coor %base(%base_shape) %c4, %c4 : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3xf64>>
  %inner = fir.array_coor %view(%inner_shape) %k : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// 2-D base with 1-D inner view: dx(1:3, 4)(k) -> dx(k, 4). Inner index
// arithmetic folds away as in test31.
// CHECK-LABEL:   func.func @test32_merge_array_coor_2d_inner_1d(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<?x?xf64>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index,
// CHECK-SAME:                      %[[VAL_3:.*]]: index) -> !fir.ref<f64> {
// CHECK:           %[[VAL_4:.*]] = arith.constant 4 : index
// CHECK:           %[[VAL_5:.*]] = fir.shape %[[VAL_1]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_6:.*]] = fir.array_coor %[[VAL_0]](%[[VAL_5]]) %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
// CHECK:           return %[[VAL_6]] : !fir.ref<f64>
// CHECK-NOT:       fir.convert
func.func @test32_merge_array_coor_2d_inner_1d(%base: !fir.ref<!fir.array<?x?xf64>>,
                                               %d1: index, %d2: index,
                                               %k: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %base_shape = fir.shape %d1, %d2 : (index, index) -> !fir.shape<2>
  %inner_shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %outer = fir.array_coor %base(%base_shape) %c1, %c4 : (!fir.ref<!fir.array<?x?xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3xf64>>
  %inner = fir.array_coor %view(%inner_shape) %k : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// 3-D base with 2-D inner view: dx(1:3, 1:4, 5)(i, j) -> dx(i, j, 5).
// CHECK-LABEL:   func.func @test33_merge_array_coor_3d_inner_2d(
// CHECK-SAME:        %[[ARG0:[A-Za-z0-9_]+]]: !fir.ref<!fir.array<?x?x?xf64>>,
// CHECK-SAME:        %[[ARG1:[A-Za-z0-9_]+]]: index, %[[ARG2:[A-Za-z0-9_]+]]: index, %[[ARG3:[A-Za-z0-9_]+]]: index,
// CHECK-SAME:        %[[ARG4:[A-Za-z0-9_]+]]: index, %[[ARG5:[A-Za-z0-9_]+]]: index)
// CHECK:           %[[C5:.*]] = arith.constant 5 : index
// CHECK:           %[[SH:.*]] = fir.shape %[[ARG1]], %[[ARG2]], %[[ARG3]]
// CHECK:           %[[AC:.*]] = fir.array_coor %[[ARG0]](%[[SH]]) %[[ARG4]], %[[ARG5]], %[[C5]]
// CHECK:           return %[[AC]] : !fir.ref<f64>
func.func @test33_merge_array_coor_3d_inner_2d(%base: !fir.ref<!fir.array<?x?x?xf64>>,
                                               %d1: index, %d2: index, %d3: index,
                                               %i: index, %j: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c5 = arith.constant 5 : index
  %base_shape = fir.shape %d1, %d2, %d3 : (index, index, index) -> !fir.shape<3>
  %inner_shape = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %outer = fir.array_coor %base(%base_shape) %c1, %c1, %c5 : (!fir.ref<!fir.array<?x?x?xf64>>, !fir.shape<3>, index, index, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3x4xf64>>
  %inner = fir.array_coor %view(%inner_shape) %i, %j : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// No fold: inner shape carries a shift (non-default lower bound).
// CHECK-LABEL:   func.func @test34_merge_no_fold_inner_shape_shift(
// CHECK:           fir.array_coor %{{.*}} -> !fir.ref<f64>
// CHECK:           fir.convert
// CHECK:           fir.array_coor %{{.*}} : (!fir.ref<!fir.array<3xf64>>, !fir.shapeshift<1>, index) -> !fir.ref<f64>
func.func @test34_merge_no_fold_inner_shape_shift(%base: !fir.ref<!fir.array<?xf64>>,
                                                  %n: index,
                                                  %k: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %base_shape = fir.shape %n : (index) -> !fir.shape<1>
  %inner_shape = fir.shape_shift %c1, %c3 : (index, index) -> !fir.shapeshift<1>
  %outer = fir.array_coor %base(%base_shape) %c1 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3xf64>>
  %inner = fir.array_coor %view(%inner_shape) %k : (!fir.ref<!fir.array<3xf64>>, !fir.shapeshift<1>, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// No fold: inner has a slice.
// CHECK-LABEL:   func.func @test35_merge_no_fold_inner_slice(
// CHECK:           fir.array_coor %{{.*}} -> !fir.ref<f64>
// CHECK:           fir.convert
// CHECK:           fir.array_coor %{{.*}} {{\[}}%{{.*}}]
func.func @test35_merge_no_fold_inner_slice(%base: !fir.ref<!fir.array<?xf64>>,
                                            %n: index,
                                            %k: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %base_shape = fir.shape %n : (index) -> !fir.shape<1>
  %inner_shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %slice = fir.slice %c1, %c3, %c1 : (index, index, index) -> !fir.slice<1>
  %outer = fir.array_coor %base(%base_shape) %c1 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3xf64>>
  %inner = fir.array_coor %view(%inner_shape) [%slice] %k : (!fir.ref<!fir.array<3xf64>>, !fir.shape<1>, !fir.slice<1>, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}

// No fold: convert reinterprets to a different element type.
// CHECK-LABEL:   func.func @test36_merge_no_fold_element_type_mismatch(
// CHECK:           fir.array_coor %{{.*}} -> !fir.ref<f64>
// CHECK:           fir.convert
// CHECK:           fir.array_coor %{{.*}} : (!fir.ref<!fir.array<6xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
func.func @test36_merge_no_fold_element_type_mismatch(%base: !fir.ref<!fir.array<?xf64>>,
                                                      %n: index,
                                                      %k: index) -> !fir.ref<i32> {
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  %base_shape = fir.shape %n : (index) -> !fir.shape<1>
  %inner_shape = fir.shape %c6 : (index) -> !fir.shape<1>
  %outer = fir.array_coor %base(%base_shape) %c1 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<6xi32>>
  %inner = fir.array_coor %view(%inner_shape) %k : (!fir.ref<!fir.array<6xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  return %inner : !fir.ref<i32>
}

// No fold: inner rank exceeds outer rank.
// CHECK-LABEL:   func.func @test37_merge_no_fold_inner_rank_too_large(
// CHECK:           fir.array_coor %{{.*}} -> !fir.ref<f64>
// CHECK:           fir.convert
// CHECK:           fir.array_coor %{{.*}} : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
func.func @test37_merge_no_fold_inner_rank_too_large(%base: !fir.ref<!fir.array<?xf64>>,
                                                     %n: index,
                                                     %i: index,
                                                     %j: index) -> !fir.ref<f64> {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %base_shape = fir.shape %n : (index) -> !fir.shape<1>
  %inner_shape = fir.shape %c3, %c4 : (index, index) -> !fir.shape<2>
  %outer = fir.array_coor %base(%base_shape) %c1 : (!fir.ref<!fir.array<?xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %view = fir.convert %outer : (!fir.ref<f64>) -> !fir.ref<!fir.array<3x4xf64>>
  %inner = fir.array_coor %view(%inner_shape) %i, %j : (!fir.ref<!fir.array<3x4xf64>>, !fir.shape<2>, index, index) -> !fir.ref<f64>
  return %inner : !fir.ref<f64>
}
