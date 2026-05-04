! Test lowering of intrinsic elemental procedure reference to HLFIR
! The goal here is not to test every intrinsics, it is to test the
! lowering framework for elemental intrinsics. This test various
! intrinsics that have different number or arguments and argument types.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

subroutine simple_elemental(x,y)
  real :: x(100), y(100)
  x = acos(y)
end subroutine
! CHECK-LABEL: func.func @_QPsimple_elemental(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_3:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_6:[a-z0-9]*]])  {{.*}}Ey
! CHECK:  %[[VAL_8:.*]] = hlfir.elemental %[[VAL_6]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xf32> {
! CHECK:  ^bb0(%[[VAL_9:.*]]: index):
! CHECK:    %[[VAL_10:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_11:.*]] = fir.load %[[VAL_10]] : !fir.ref<f32>
! CHECK:    %[[VAL_12:.*]] = math.acos %[[VAL_11]] fastmath<contract> : f32
! CHECK:    hlfir.yield_element %[[VAL_12]] : f32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_8]]

subroutine elemental_mixed_args(x,y, scalar)
  real :: x(100), y(100), scalar
  x = atan2(x, scalar)
end subroutine
! CHECK-LABEL: func.func @_QPelemental_mixed_args(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]]  {{.*}}Escalar
! CHECK:  %[[VAL_4:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]](%[[VAL_5:[a-z0-9]*]])  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_8:.*]] = fir.shape %[[VAL_7]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]](%[[VAL_8:[a-z0-9]*]])  {{.*}}Ey
! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<f32>
! CHECK:  %[[VAL_11:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xf32> {
! CHECK:  ^bb0(%[[VAL_12:.*]]: index):
! CHECK:    %[[VAL_13:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_12]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<f32>
! CHECK:    %[[VAL_15:.*]] = math.atan2 %[[VAL_14]], %[[VAL_10]] fastmath<contract> : f32
! CHECK:    hlfir.yield_element %[[VAL_15]] : f32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_11]]

subroutine elemental_assumed_shape_arg(x)
  real :: x(:)
  print *, sin(x)
end subroutine
! CHECK-LABEL: func.func @_QPelemental_assumed_shape_arg(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0:[a-z0-9]*]]  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_8:.*]]:3 = fir.box_dims %[[VAL_1]]#0, %[[VAL_7]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.shape %[[VAL_8]]#1 : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_10:.*]] = hlfir.elemental %[[VAL_9]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:  ^bb0(%[[VAL_11:.*]]: index):
! CHECK:    %[[VAL_12:.*]] = hlfir.designate %[[VAL_1]]#0 (%[[VAL_11]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_12]] : !fir.ref<f32>
! CHECK:    %[[VAL_14:.*]] = math.sin %[[VAL_13]] fastmath<contract> : f32
! CHECK:    hlfir.yield_element %[[VAL_14]] : f32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_10]]

subroutine elemental_with_char_args(x,y)
  character(*) :: x(100), y(:)
  print *, scan(x, y)
end subroutine
! CHECK-LABEL: func.func @_QPelemental_with_char_args(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3:[a-z0-9]*]](%[[VAL_5:[a-z0-9]*]]) typeparams %[[VAL_2:[a-z0-9]*]]#1  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_1:[a-z0-9]*]]  {{.*}}Ey
! CHECK:  %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] unordered : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_14:.*]]: index):
! CHECK:    %[[VAL_15:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_14]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_18:.*]]:2 = fir.unboxchar %[[VAL_15]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:    %[[VAL_16:.*]] = fir.box_elesize %[[VAL_7]]#1 : (!fir.box<!fir.array<?x!fir.char<1,?>>>) -> index
! CHECK:    %[[VAL_17:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_14]])  typeparams %[[VAL_16]] : (!fir.box<!fir.array<?x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_19:.*]]:2 = fir.unboxchar %[[VAL_17]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:    %[[VAL_20:.*]] = arith.constant false
! CHECK:    %[[VAL_21:.*]] = fir.convert %[[VAL_18]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    %[[VAL_22:.*]] = fir.convert %[[VAL_2]]#1 : (index) -> i64
! CHECK:    %[[VAL_23:.*]] = fir.convert %[[VAL_19]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:    %[[VAL_24:.*]] = fir.convert %[[VAL_16]] : (index) -> i64
! CHECK:    %[[VAL_25:.*]] = fir.call @_FortranAScan1(%[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]], %[[VAL_20]])
! CHECK:    %[[VAL_26:.*]] = fir.convert %[[VAL_25]] : (i64) -> i32
! CHECK:    hlfir.yield_element %[[VAL_26]] : i32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_13]]


! -----------------------------------------------------------------------------
!  Test elemental character intrinsics with non compile time constant result
!  length.
! -----------------------------------------------------------------------------

subroutine test_adjustl(x)
  character(*) :: x(100)
  call bar(adjustl(x))
end subroutine
! CHECK-LABEL: func.func @_QPtest_adjustl(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3:.*]](%[[VAL_5:[a-z0-9]*]]) typeparams %[[VAL_2:[a-z0-9]*]]#1  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_2]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    fir.call @_FortranAAdjustl
! CHECK:    %[[VAL_24:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_22:.*]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
! CHECK:    %[[VAL_25:.*]] = arith.constant true
! CHECK:    %[[VAL_26:.*]] = hlfir.as_expr %[[VAL_24]]#0 move %[[VAL_25]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    hlfir.yield_element %[[VAL_26]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  }

subroutine test_adjustr(x)
  character(*) :: x(100)
  call bar(adjustr(x))
end subroutine
! CHECK-LABEL: func.func @_QPtest_adjustr(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_3:.*]](%[[VAL_5:[a-z0-9]*]]) typeparams %[[VAL_2:[a-z0-9]*]]#1  {{.*}}Ex
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_2]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    fir.call @_FortranAAdjustr
! CHECK:    %[[VAL_24:.*]]:2 = hlfir.declare %{{.*}} typeparams %[[VAL_22:.*]] {uniq_name = ".tmp.intrinsic_result"} : (!fir.heap<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.heap<!fir.char<1,?>>)
! CHECK:    %[[VAL_25:.*]] = arith.constant true
! CHECK:    %[[VAL_26:.*]] = hlfir.as_expr %[[VAL_24]]#0 move %[[VAL_25]] : (!fir.boxchar<1>, i1) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    hlfir.yield_element %[[VAL_26]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  }

subroutine test_merge(x, y, mask)
  character(*) :: x(100), y(100)
  logical :: mask(100)
  call bar(merge(x, y, mask))
end subroutine
! CHECK-LABEL: func.func @_QPtest_merge(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_2:[a-z0-9]*]](%[[VAL_4:[a-z0-9]*]])  {{.*}}Emask
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_7:[a-z0-9]*]](%[[VAL_9:[a-z0-9]*]]) typeparams %[[VAL_6:[a-z0-9]*]]#1  {{.*}}Ex
! CHECK:  %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_12:[a-z0-9]*]](%[[VAL_14:[a-z0-9]*]]) typeparams %[[VAL_11:[a-z0-9]*]]#1  {{.*}}Ey
! CHECK:  %[[VAL_16:.*]] = hlfir.elemental %[[VAL_9]] typeparams %[[VAL_6]]#1 unordered : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_17:.*]]: index):
! CHECK:    %[[VAL_18:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_17]])  typeparams %[[VAL_6]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_21:.*]]:2 = fir.unboxchar %[[VAL_18]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:    %[[VAL_19:.*]] = hlfir.designate %[[VAL_15]]#0 (%[[VAL_17]])  typeparams %[[VAL_11]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_22:.*]]:2 = fir.unboxchar %[[VAL_19]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:    %[[VAL_20:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_17]])  : (!fir.ref<!fir.array<100x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_23:.*]] = fir.load %[[VAL_20]] : !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_24:.*]] = fir.convert %[[VAL_23]] : (!fir.logical<4>) -> i1
! CHECK:    %[[VAL_25:.*]] = arith.select %[[VAL_24]], %[[VAL_21]]#0, %[[VAL_22]]#0 : !fir.ref<!fir.char<1,?>>
! CHECK:    %[[VAL_26:.*]]:2 = hlfir.declare %[[VAL_25]] typeparams %[[VAL_6]]#1 {uniq_name = ".tmp.intrinsic_result"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
! CHECK:    %[[VAL_27:.*]] = hlfir.as_expr %[[VAL_26]]#0 : (!fir.boxchar<1>) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    hlfir.yield_element %[[VAL_27]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  }
