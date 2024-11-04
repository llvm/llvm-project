// RUN: fir-opt %s -opt-bufferization | FileCheck %s

func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.logical<4> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.logical<4> {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFFtestEtest"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %5:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.load %2#0 : !fir.ref<i32>
  %7 = fir.convert %6 : (i32) -> i64
  %8 = fir.shape %c7 : (index) -> !fir.shape<1>
  %9 = hlfir.designate %1#0 (%7, %c1:%c7:%c1)  shape %8 : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
  %10 = fir.load %5#0 : !fir.ref<i32>
  %11 = hlfir.elemental %8 unordered : (!fir.shape<1>) -> !hlfir.expr<7x!fir.logical<4>> {
  ^bb0(%arg3: index):
    %14 = hlfir.designate %9 (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
    %15 = fir.load %14 : !fir.ref<i32>
    %16 = arith.cmpi sge, %15, %10 : i32
    %17 = fir.convert %16 : (i1) -> !fir.logical<4>
    hlfir.yield_element %17 : !fir.logical<4>
  }
  %12 = hlfir.any %11 : (!hlfir.expr<7x!fir.logical<4>>) -> !fir.logical<4>
  hlfir.assign %12 to %4#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  hlfir.destroy %11 : !hlfir.expr<7x!fir.logical<4>>
  %13 = fir.load %4#1 : !fir.ref<!fir.logical<4>>
  return %13 : !fir.logical<4>
}
// CHECK-LABEL:  func.func @_QFPtest(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.logical<4> {
// CHECK-NEXT:     %false = arith.constant false
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c4 = arith.constant 4 : index
// CHECK-NEXT:     %c7 = arith.constant 7 : index
// CHECK-NEXT:     %[[V0:.*]] = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
// CHECK-NEXT:     %[[V1:.*]]:2 = hlfir.declare %arg0(%[[V0]])
// CHECK-NEXT:     %[[V2:.*]]:2 = hlfir.declare %arg1
// CHECK-NEXT:     %[[V3:.*]] = fir.alloca !fir.logical<4>
// CHECK-NEXT:     %[[V4:.*]]:2 = hlfir.declare %[[V3]]
// CHECK-NEXT:     %[[V5:.*]]:2 = hlfir.declare %arg2
// CHECK-NEXT:     %[[V6:.*]] = fir.load %[[V2]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V7:.*]] = fir.convert %[[V6]] : (i32) -> i64
// CHECK-NEXT:     %[[V8:.*]] = fir.shape %c7 : (index) -> !fir.shape<1>
// CHECK-NEXT:     %[[V9:.*]] = hlfir.designate %[[V1]]#0 (%[[V7]], %c1:%c7:%c1)  shape %[[V8]] : (!fir.ref<!fir.array<4x7xi32>>, i64, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<7xi32>>
// CHECK-NEXT:     %[[V10:.*]] = fir.load %[[V5]]#0 : !fir.ref<i32>
// CHECK-NEXT:     %[[V11:.*]] = fir.do_loop %arg3 = %c1 to %c7 step %c1 iter_args(%arg4 = %false) -> (i1) {
// CHECK-NEXT:       %[[V14:.*]] = hlfir.designate %[[V9]] (%arg3)  : (!fir.box<!fir.array<7xi32>>, index) -> !fir.ref<i32>
// CHECK-NEXT:       %[[V15:.*]] = fir.load %[[V14]] : !fir.ref<i32>
// CHECK-NEXT:       %[[V16:.*]] = arith.cmpi sge, %[[V15]], %[[V10]] : i32
// CHECK-NEXT:       %[[V17:.*]] = arith.ori %arg4, %[[V16]] : i1
// CHECK-NEXT:       fir.result %[[V17]] : i1
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V12:.*]] = fir.convert %[[V11]] : (i1) -> !fir.logical<4>
// CHECK-NEXT:     hlfir.assign %[[V12]] to %[[V4]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
// CHECK-NEXT:     %[[V13:.*]] = fir.load %[[V4]]#1 : !fir.ref<!fir.logical<4>>
// CHECK-NEXT:     return %[[V13]] : !fir.logical<4>


func.func @_QFPtest_dim(%arg0: !fir.ref<!fir.array<4x7xi32>> {fir.bindc_name = "b"}, %arg1: !fir.ref<i32> {fir.bindc_name = "row"}, %arg2: !fir.ref<i32> {fir.bindc_name = "val"}) -> !fir.array<4x!fir.logical<4>> {
  %c2_i32 = arith.constant 2 : i32
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c7 = arith.constant 7 : index
  %0 = fir.shape %c4, %c7 : (index, index) -> !fir.shape<2>
  %1:2 = hlfir.declare %arg0(%0) {uniq_name = "_QFFtestEb"} : (!fir.ref<!fir.array<4x7xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<4x7xi32>>, !fir.ref<!fir.array<4x7xi32>>)
  %2:2 = hlfir.declare %arg1 {uniq_name = "_QFFtestErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %3 = fir.alloca !fir.array<4x!fir.logical<4>> {bindc_name = "test", uniq_name = "_QFFtestEtest"}
  %4 = fir.shape %c4 : (index) -> !fir.shape<1>
  %5:2 = hlfir.declare %3(%4) {uniq_name = "_QFFtestEtest"} : (!fir.ref<!fir.array<4x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<4x!fir.logical<4>>>, !fir.ref<!fir.array<4x!fir.logical<4>>>)
  %6:2 = hlfir.declare %arg2 {uniq_name = "_QFFtestEval"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7 = hlfir.designate %1#0 (%c1:%c4:%c1, %c1:%c7:%c1)  shape %0 : (!fir.ref<!fir.array<4x7xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<4x7xi32>>
  %8 = fir.load %6#0 : !fir.ref<i32>
  %9 = hlfir.elemental %0 unordered : (!fir.shape<2>) -> !hlfir.expr<4x7x!fir.logical<4>> {
  ^bb0(%arg3: index, %arg4: index):
    %12 = hlfir.designate %7 (%arg3, %arg4)  : (!fir.ref<!fir.array<4x7xi32>>, index, index) -> !fir.ref<i32>
    %13 = fir.load %12 : !fir.ref<i32>
    %14 = arith.cmpi sge, %13, %8 : i32
    %15 = fir.convert %14 : (i1) -> !fir.logical<4>
    hlfir.yield_element %15 : !fir.logical<4>
  }
  %10 = hlfir.any %9 dim %c2_i32 : (!hlfir.expr<4x7x!fir.logical<4>>, i32) -> !hlfir.expr<4x!fir.logical<4>>
  hlfir.assign %10 to %5#0 : !hlfir.expr<4x!fir.logical<4>>, !fir.ref<!fir.array<4x!fir.logical<4>>>
  hlfir.destroy %10 : !hlfir.expr<4x!fir.logical<4>>
  hlfir.destroy %9 : !hlfir.expr<4x7x!fir.logical<4>>
  %11 = fir.load %5#1 : !fir.ref<!fir.array<4x!fir.logical<4>>>
  return %11 : !fir.array<4x!fir.logical<4>>
}
// CHECK-LABEL:  func.func @_QFPtest_dim(
// CHECK: {{.*}} = hlfir.any {{.*}} dim %c2_i32


func.func @_Qtest_recursive() attributes {fir.bindc_name = "test"} {
  %c1 = arith.constant 1 : index
  %true = arith.constant true
  %false = arith.constant false
  %c0_i64 = arith.constant 0 : i64
  %c2_i32 = arith.constant 2 : i32
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %1:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
  %2 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
  %3:2 = hlfir.declare %2 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %4 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFEn"}
  %5:2 = hlfir.declare %4 {uniq_name = "_QFEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %6 = fir.alloca !fir.array<1x!fir.logical<4>> {bindc_name = "ra", uniq_name = "_QFEra"}
  %7 = fir.shape %c1 : (index) -> !fir.shape<1>
  %8:2 = hlfir.declare %6(%7) {uniq_name = "_QFEra"} : (!fir.ref<!fir.array<1x!fir.logical<4>>>, !fir.shape<1>) -> (!fir.ref<!fir.array<1x!fir.logical<4>>>, !fir.ref<!fir.array<1x!fir.logical<4>>>)
  %9 = fir.alloca !fir.logical<4> {bindc_name = "rs", uniq_name = "_QFErs"}
  %10:2 = hlfir.declare %9 {uniq_name = "_QFErs"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  %11 = fir.allocmem !fir.array<?xi32>, %c1 {fir.must_be_heap = true, uniq_name = "_QFEa.alloc"}
  %12 = fir.embox %11(%7) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
  fir.store %12 to %1#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  hlfir.assign %c1_i32 to %5#0 : i32, !fir.ref<i32>
  %13 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %14:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  fir.do_loop %arg0 = %c1 to %14#1 step %c1 unordered {
    %27:3 = fir.box_dims %13, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %28 = arith.subi %27#0, %c1 : index
    %29 = arith.addi %arg0, %28 : index
    %30 = hlfir.designate %13 (%29)  : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
    hlfir.assign %c2_i32 to %30 : i32, !fir.ref<i32>
  }
  %15 = fir.load %5#0 : !fir.ref<i32>
  %16 = fir.convert %15 : (i32) -> i64
  %17 = arith.cmpi sgt, %16, %c0_i64 : i64
  %18 = arith.select %17, %16, %c0_i64 : i64
  %19 = fir.convert %18 : (i64) -> index
  %20 = fir.shape %19 : (index) -> !fir.shape<1>
  %21 = hlfir.elemental %20 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
  ^bb0(%arg0: index):
    %27 = fir.load %1#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %28:3 = fir.box_dims %27, %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %29 = arith.addi %28#0, %28#1 : index
    %30 = arith.subi %29, %c1 : index
    %31 = arith.subi %30, %28#0 : index
    %32 = arith.addi %31, %c1 : index
    %33 = arith.cmpi sgt, %32, %c0 : index
    %34 = arith.select %33, %32, %c0 : index
    %35 = fir.shape %34 : (index) -> !fir.shape<1>
    %36 = hlfir.designate %27 (%28#0:%30:%c1)  shape %35 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
    %37 = hlfir.elemental %35 unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
    ^bb0(%arg1: index):
      %39 = hlfir.designate %36 (%arg1)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
      %40 = fir.load %39 : !fir.ref<i32>
      %41 = arith.cmpi eq, %40, %c1_i32 : i32
      %42 = fir.convert %41 : (i1) -> !fir.logical<4>
      hlfir.yield_element %42 : !fir.logical<4>
    }
    %38 = hlfir.any %37 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
    hlfir.destroy %37 : !hlfir.expr<?x!fir.logical<4>>
    hlfir.yield_element %38 : !fir.logical<4>
  }
  %22 = hlfir.any %21 : (!hlfir.expr<?x!fir.logical<4>>) -> !fir.logical<4>
  hlfir.assign %22 to %10#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
  hlfir.destroy %21 : !hlfir.expr<?x!fir.logical<4>>
  %23 = fir.load %10#0 : !fir.ref<!fir.logical<4>>
  %24 = fir.convert %23 : (!fir.logical<4>) -> i1
  %25 = arith.xori %24, %true : i1
  cf.cond_br %25, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  %26 = fir.call @_FortranAStopStatement(%c2_i32, %false, %false) fastmath<contract> : (i32, i1, i1) -> none
  fir.unreachable
^bb2:  // pred: ^bb0
  return
}
// CHECK-LABEL: func.func @_Qtest_recursive()
// CHECK:    %[[V20:.*]] = fir.do_loop %arg0 = %c1 to %{{.*}} step %c1 iter_args(%arg1 = %false) -> (i1) {
// CHECK:      %[[V26:.*]] = fir.load %[[V1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:      %[[V27:.*]]:3 = fir.box_dims %[[V26]], %c0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
// CHECK:      %[[V28:.*]] = arith.addi %[[V27]]#0, %[[V27]]#1 : index
// CHECK:      %[[V29:.*]] = arith.subi %[[V28]], %c1 : index
// CHECK:      %[[V30:.*]] = arith.subi %[[V29]], %[[V27]]#0 : index
// CHECK:      %[[V31:.*]] = arith.addi %[[V30]], %c1 : index
// CHECK:      %[[V32:.*]] = arith.cmpi sgt, %[[V31]], %c0 : index
// CHECK:      %[[V33:.*]] = arith.select %[[V32]], %[[V31]], %c0 : index
// CHECK:      %[[V34:.*]] = fir.shape %[[V33]] : (index) -> !fir.shape<1>
// CHECK:      %[[V35:.*]] = hlfir.designate %[[V26]] (%[[V27]]#0:%[[V29]]:%c1)  shape %[[V34]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
// CHECK:      %[[V36:.*]] = fir.do_loop %arg2 = %c1 to %[[V33]] step %c1 iter_args(%arg3 = %false) -> (i1) {
// CHECK:        %[[V38:.*]] = hlfir.designate %[[V35]] (%arg2)  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
// CHECK:        %[[V39:.*]] = fir.load %[[V38]] : !fir.ref<i32>
// CHECK:        %[[V40:.*]] = arith.cmpi eq, %[[V39]], %c1_i32 : i32
// CHECK:        %[[V41:.*]] = arith.ori %arg3, %[[V40]] : i1
// CHECK:        fir.result %[[V41]] : i1
// CHECK:      }
// CHECK:      %[[V37:.*]] = arith.ori %arg1, %[[V36]] : i1
// CHECK:      fir.result %[[V37]] : i1
// CHECK:    }
