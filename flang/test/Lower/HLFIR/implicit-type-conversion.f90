! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:   func.func @_QPtest1(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest1Ex"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest1Ey"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.logical<4>) -> i32
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }
subroutine test1(x, y)
  integer :: x
  logical :: y
  x = y
end subroutine test1

! CHECK-LABEL:   func.func @_QPtest2(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest2Ex"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest2Ey"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_2]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i32) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_3]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           return
! CHECK:         }
subroutine test2(x, y)
  integer :: x
  logical :: y
  y = x
end subroutine test2

! CHECK-LABEL:   func.func @_QPtest3(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest3Ex"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest3Ey"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.no_reassoc %[[VAL_6]] : i1
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[VAL_8]] to %[[VAL_2]]#0 : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           return
! CHECK:         }
subroutine test3(x, y)
  logical :: x
  integer :: y
  x = (y.eq.1)
end subroutine test3

! CHECK-LABEL:   func.func @_QPtest4(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest4Ex"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest4Ey"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i32
! CHECK:           %[[VAL_6:.*]] = arith.cmpi eq, %[[VAL_4]], %[[VAL_5]] : i32
! CHECK:           %[[VAL_7:.*]] = hlfir.no_reassoc %[[VAL_6]] : i1
! CHECK:           %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (i1) -> i32
! CHECK:           hlfir.assign %[[VAL_8]] to %[[VAL_2]]#0 : i32, !fir.ref<i32>
! CHECK:           return
! CHECK:         }
subroutine test4(x, y)
  integer :: x
  integer :: y
  x = (y.eq.1)
end subroutine test4

! CHECK-LABEL:   func.func @_QPtest5(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest5Ex"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest5Ey"} : (!fir.ref<!fir.logical<4>>, !fir.dscope) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.logical<4>) -> i32
! CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_2]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           return
! CHECK:         }
subroutine test5(x, y)
  integer :: x(:)
  logical :: y
  x = y
end subroutine test5

! CHECK-LABEL:   func.func @_QPtest6(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest6Ex"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest6Ey"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:           ^bb0(%[[VAL_8:.*]]: index):
! CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_8]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<!fir.logical<4>>
! CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.logical<4>) -> i32
! CHECK:             hlfir.yield_element %[[VAL_11]] : i32
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12:.*]] to %[[VAL_2]]#0 : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<?xi32>
! CHECK:           return
! CHECK:         }
subroutine test6(x, y)
  integer :: x(:)
  logical :: y(:)
  x = y
end subroutine test6

! CHECK-LABEL:   func.func @_QPtest7(
! CHECK-SAME:                        %[[VAL_0:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>> {fir.bindc_name = "x"},
! CHECK-SAME:                        %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"}) {
! CHECK:           %[[VAL_2:.*]]:2 = hlfir.declare %[[VAL_0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest7Ex"} : (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.dscope) -> (!fir.box<!fir.array<?x!fir.logical<4>>>, !fir.box<!fir.array<?x!fir.logical<4>>>)
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFtest7Ey"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:           %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]]#0, %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[VAL_6:.*]] = fir.shape %[[VAL_5]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[VAL_7:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:           ^bb0(%[[VAL_8:.*]]: index):
! CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_8]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:             %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (i32) -> !fir.logical<4>
! CHECK:             hlfir.yield_element %[[VAL_11]] : !fir.logical<4>
! CHECK:           }
! CHECK:           hlfir.assign %[[VAL_12:.*]] to %[[VAL_2]]#0 : !hlfir.expr<?x!fir.logical<4>>, !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<?x!fir.logical<4>>
! CHECK:           return
! CHECK:         }
subroutine test7(x, y)
  logical :: x(:)
  integer :: y(:)
  x = y
end subroutine test7
