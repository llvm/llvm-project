// Test optimized bufferization for hlfir.assign with scalar RHS.
// RUN: fir-opt --opt-bufferization %s | FileCheck %s

func.func @_QPtest1() {
  %cst = arith.constant 0.000000e+00 : f32
  %c11 = arith.constant 11 : index
  %c13 = arith.constant 13 : index
  %0 = fir.alloca !fir.array<11x13xf32> {bindc_name = "x", uniq_name = "_QFtest1Ex"}
  %1 = fir.shape %c11, %c13 : (index, index) -> !fir.shape<2>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<11x13xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<11x13xf32>>, !fir.ref<!fir.array<11x13xf32>>)
  hlfir.assign %cst to %2#0 : f32, !fir.ref<!fir.array<11x13xf32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest1() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 143 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 11 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 13 : index
// CHECK:           %[[VAL_5:.*]] = fir.alloca !fir.array<11x13xf32> {bindc_name = "x", uniq_name = "_QFtest1Ex"}
// CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_3]], %[[VAL_4]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_5]](%[[VAL_6]]) {uniq_name = "_QFtest1Ex"} : (!fir.ref<!fir.array<11x13xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<11x13xf32>>, !fir.ref<!fir.array<11x13xf32>>)
// CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.array<11x13xf32>>) -> !fir.ref<!fir.array<143xf32>>
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_0]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_8]] (%[[VAL_9]]) : (!fir.ref<!fir.array<143xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_2]] to %[[VAL_10]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest2(%arg0: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
  hlfir.assign %c0_i32 to %0#0 : i32, !fir.box<!fir.array<?x?xi32>>
  return
}
// CHECK-LABEL:   func.func @_QPtest2(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x?xi32>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest2Ex"} : (!fir.box<!fir.array<?x?xi32>>) -> (!fir.box<!fir.array<?x?xi32>>, !fir.box<!fir.array<?x?xi32>>)
// CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_2]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_4]]#0, %[[VAL_1]] : (!fir.box<!fir.array<?x?xi32>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_6]]#1 step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_5]]#1 step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]], %[[VAL_7]])  : (!fir.box<!fir.array<?x?xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_3]] to %[[VAL_9]] : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest4(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "x"}) {
  %true = arith.constant true
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
  %1 = fir.convert %true : (i1) -> !fir.logical<4>
  %2 = fir.load %0#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
  hlfir.assign %1 to %2 : !fir.logical<4>, !fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest4(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant true
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFtest4Ex"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_3]] : (i1) -> !fir.logical<4>
// CHECK:           %[[VAL_6:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>>
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_8:.*]] = %[[VAL_1]] to %[[VAL_7]]#1 step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_2]] : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]]#0, %[[VAL_1]] : index
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_8]], %[[VAL_10]] : index
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_6]] (%[[VAL_11]])  : (!fir.box<!fir.ptr<!fir.array<?x!fir.logical<4>>>>, index) -> !fir.ref<!fir.logical<4>>
// CHECK:             hlfir.assign %[[VAL_5]] to %[[VAL_12]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest3(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  hlfir.assign %c0_i32 to %0#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest3(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_6]]#1 step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]]#0, %[[VAL_1]] : index
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
// CHECK:             %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]] (%[[VAL_10]])  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_3]] to %[[VAL_11]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest5(%arg0: !fir.ref<!fir.array<77xcomplex<f32>>> {fir.bindc_name = "x"}) {
  %cst = arith.constant 0.000000e+00 : f32
  %c77 = arith.constant 77 : index
  %0 = fir.shape %c77 : (index) -> !fir.shape<1>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<77xcomplex<f32>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<77xcomplex<f32>>>, !fir.ref<!fir.array<77xcomplex<f32>>>)
  %2 = fir.undefined complex<f32>
  %3 = fir.insert_value %2, %cst, [0 : index] : (complex<f32>, f32) -> complex<f32>
  %4 = fir.insert_value %3, %cst, [1 : index] : (complex<f32>, f32) -> complex<f32>
  hlfir.assign %4 to %1#0 : complex<f32>, !fir.ref<!fir.array<77xcomplex<f32>>>
  return
}
// CHECK-LABEL:   func.func @_QPtest5(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<77xcomplex<f32>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 77 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "_QFtest5Ex"} : (!fir.ref<!fir.array<77xcomplex<f32>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<77xcomplex<f32>>>, !fir.ref<!fir.array<77xcomplex<f32>>>)
// CHECK:           %[[VAL_6:.*]] = fir.undefined complex<f32>
// CHECK:           %[[VAL_7:.*]] = fir.insert_value %[[VAL_6]], %[[VAL_2]], [0 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           %[[VAL_8:.*]] = fir.insert_value %[[VAL_7]], %[[VAL_2]], [1 : index] : (complex<f32>, f32) -> complex<f32>
// CHECK:           fir.do_loop %[[VAL_9:.*]] = %[[VAL_1]] to %[[VAL_3]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<77xcomplex<f32>>>, index) -> !fir.ref<complex<f32>>
// CHECK:             hlfir.assign %[[VAL_8]] to %[[VAL_10]] : complex<f32>, !fir.ref<complex<f32>>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QPtest6(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "x"}) {
  %c0_i32 = arith.constant 0 : i32
  %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
  hlfir.assign %c0_i32 to %0#0 realloc : i32, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
  return
}

// CHECK-LABEL:   func.func @_QPtest6(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>> {fir.bindc_name = "x"}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest6Ex"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>)
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xi32>>>>
// CHECK:           %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_2]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_5]], %[[VAL_1]] : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, index) -> (index, index, index)
// CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]]#1, %[[VAL_7]]#1 : index
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_8]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_10:.*]] = fir.rebox %[[VAL_5]](%[[VAL_9]]) : (!fir.box<!fir.heap<!fir.array<?x?xi32>>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:           fir.do_loop %[[VAL_11:.*]] = %[[VAL_1]] to %[[VAL_8]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_12:.*]] = hlfir.designate %[[VAL_10]] (%[[VAL_11]]) : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_3]] to %[[VAL_12]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

func.func @_QQmain() {
  return
}

func.func private @_QFPtest7(%arg0: !fir.ref<tuple<!fir.box<!fir.array<2x2xi32>>>> {fir.host_assoc}) attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.dummy_scope : !fir.dscope
  %c0_i32 = arith.constant 0 : i32
  %1 = fir.coordinate_of %arg0, %c0_i32 : (!fir.ref<tuple<!fir.box<!fir.array<2x2xi32>>>>, i32) -> !fir.ref<!fir.box<!fir.array<2x2xi32>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.array<2x2xi32>>>
  %3 = fir.box_addr %2 : (!fir.box<!fir.array<2x2xi32>>) -> !fir.ref<!fir.array<2x2xi32>>
  %c0 = arith.constant 0 : index
  %4:3 = fir.box_dims %2, %c0 : (!fir.box<!fir.array<2x2xi32>>, index) -> (index, index, index)
  %c1 = arith.constant 1 : index
  %5:3 = fir.box_dims %2, %c1 : (!fir.box<!fir.array<2x2xi32>>, index) -> (index, index, index)
  %6 = fir.shape %4#1, %5#1 : (index, index) -> !fir.shape<2>
  %7:2 = hlfir.declare %3(%6) {fortran_attrs = #fir.var_attrs<host_assoc>, uniq_name = "_QFEa"} : (!fir.ref<!fir.array<2x2xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<2x2xi32>>, !fir.ref<!fir.array<2x2xi32>>)
  %c0_i32_0 = arith.constant 0 : i32
  hlfir.assign %c0_i32_0 to %7#0 : i32, !fir.ref<!fir.array<2x2xi32>>
  return
}

// CHECK-LABEL:   func.func private @_QFPtest7({{.*}}) {{.*}} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK:           fir.do_loop %[[VAL_1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:             %[[VAL_2:.*]] = hlfir.designate %{{.*}} (%[[VAL_1]]) : (!fir.ref<!fir.array<4xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_0]] to %[[VAL_2]] : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:         }

func.func @_QPtest8() {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<120xi8> {uniq_name = "_QFtest8Ea"}
  %c0 = arith.constant 0 : index
  %2 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<120xi8>>, index) -> !fir.ref<i8>
  %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<2x10xf32>>
  %c2 = arith.constant 2 : index
  %c10 = arith.constant 10 : index
  %4 = fir.shape %c2, %c10 : (index, index) -> !fir.shape<2>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFtest8Ea"} : (!fir.ptr<!fir.array<2x10xf32>>, !fir.shape<2>) -> (!fir.ptr<!fir.array<2x10xf32>>, !fir.ptr<!fir.array<2x10xf32>>)
  %c0_0 = arith.constant 0 : index
  %6 = fir.coordinate_of %1, %c0_0 : (!fir.ref<!fir.array<120xi8>>, index) -> !fir.ref<i8>
  %7 = fir.convert %6 : (!fir.ref<i8>) -> !fir.ptr<!fir.array<30xf32>>
  %c30 = arith.constant 30 : index
  %8 = fir.shape %c30 : (index) -> !fir.shape<1>
  %9:2 = hlfir.declare %7(%8) {uniq_name = "_QFtest8Eb"} : (!fir.ptr<!fir.array<30xf32>>, !fir.shape<1>) -> (!fir.ptr<!fir.array<30xf32>>, !fir.ptr<!fir.array<30xf32>>)
  %cst = arith.constant -1.000000e+00 : f32
  hlfir.assign %cst to %5#0 : f32, !fir.ptr<!fir.array<2x10xf32>>
  return
}

// CHECK-LABEL:   func.func @_QPtest8() {
// CHECK:           %[[VAL_0:.*]] = arith.constant -1.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}}(%{{.*}}) {uniq_name = "_QFtest8Ea"} : (!fir.ptr<!fir.array<2x10xf32>>, !fir.shape<2>) -> (!fir.ptr<!fir.array<2x10xf32>>, !fir.ptr<!fir.array<2x10xf32>>)
// CHECK:           %[[VAL_2:.*]] = fir.convert %[[VAL_1]]#0 : (!fir.ptr<!fir.array<2x10xf32>>) -> !fir.ref<!fir.array<20xf32>>
// CHECK:           fir.do_loop %[[VAL_3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} unordered {
// CHECK:             %[[VAL_4:.*]] = hlfir.designate %[[VAL_2]] (%[[VAL_3]]) : (!fir.ref<!fir.array<20xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[VAL_0]] to %[[VAL_4]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:         }
