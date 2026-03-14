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
