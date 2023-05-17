! Test lowering of elemental intrinsic operations with array arguments to HLFIR
! RUN: bbc -emit-fir -hlfir -o - %s 2>&1 | FileCheck %s

subroutine binary(x, y)
  integer :: x(100), y(100)
  x = x+y
end subroutine
! CHECK-LABEL: func.func @_QPbinary(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_6:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_8:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_9:.*]]: index):
! CHECK:    %[[VAL_10:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_7]]#0 (%[[VAL_9]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_12:.*]] = fir.load %[[VAL_10]] : !fir.ref<i32>
! CHECK:    %[[VAL_13:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:    %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
! CHECK:    hlfir.yield_element %[[VAL_14]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_8]]

subroutine binary_with_scalar_and_array(x, y)
  integer :: x(100), y
  x = x+y
end subroutine
! CHECK-LABEL: func.func @_QPbinary_with_scalar_and_array(
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}} {{.*}}y
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_8]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_10:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
! CHECK:    %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_6]] : i32
! CHECK:    hlfir.yield_element %[[VAL_11]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_7]]

subroutine char_binary(x, y)
  character(*) :: x(100), y(100)
  call test_char(x//y)
end subroutine
! CHECK-LABEL: func.func @_QPchar_binary(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_5:.*]]) typeparams %[[VAL_2:.*]]#1 {{.*}}x
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_10:.*]]) typeparams %[[VAL_7:.*]]#1 {{.*}}y
! CHECK:  %[[VAL_12:.*]] = arith.addi %[[VAL_2]]#1, %[[VAL_7]]#1 : index
! CHECK:  %[[VAL_13:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_12]] : (!fir.shape<1>, index) -> !hlfir.expr<100x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_14:.*]]: index):
! CHECK:    %[[VAL_15:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_14]])  typeparams %[[VAL_2]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_16:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_14]])  typeparams %[[VAL_7]]#1 : (!fir.box<!fir.array<100x!fir.char<1,?>>>, index, index) -> !fir.boxchar<1>
! CHECK:    %[[VAL_17:.*]] = hlfir.concat %[[VAL_15]], %[[VAL_16]] len %[[VAL_12]] : (!fir.boxchar<1>, !fir.boxchar<1>, index) -> !hlfir.expr<!fir.char<1,?>>
! CHECK:    hlfir.yield_element %[[VAL_17]] : !hlfir.expr<!fir.char<1,?>>
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_13]]

subroutine unary(x, n)
  integer :: n
  logical :: x(n)
  x = .not.x
end subroutine
! CHECK-LABEL: func.func @_QPunary(
! CHECK:  %[[VAL_10:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_9:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_11:.*]] = hlfir.elemental %[[VAL_9]] : (!fir.shape<1>) -> !hlfir.expr<?x!fir.logical<4>> {
! CHECK:  ^bb0(%[[VAL_12:.*]]: index):
! CHECK:    %[[VAL_13:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_12]])  : (!fir.box<!fir.array<?x!fir.logical<4>>>, index) -> !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_13]] : !fir.ref<!fir.logical<4>>
! CHECK:    %[[VAL_15:.*]] = arith.constant true
! CHECK:    %[[VAL_16:.*]] = fir.convert %[[VAL_14]] : (!fir.logical<4>) -> i1
! CHECK:    %[[VAL_17:.*]] = arith.xori %[[VAL_16]], %[[VAL_15]] : i1
! CHECK:    %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i1) -> !fir.logical<4>
! CHECK:    hlfir.yield_element %[[VAL_18]] : !fir.logical<4>
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_11]]

subroutine char_unary(x)
  character(10) :: x(20)
  call test_char_2((x))
end subroutine
! CHECK-LABEL: func.func @_QPchar_unary(
! CHECK:  %[[VAL_6:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_5:.*]]) typeparams %[[VAL_2:[^ ]*]] {{.*}}x
! CHECK:  %[[VAL_7:.*]] = hlfir.elemental %[[VAL_5]] typeparams %[[VAL_2]] : (!fir.shape<1>, index) -> !hlfir.expr<20x!fir.char<1,?>> {
! CHECK:  ^bb0(%[[VAL_8:.*]]: index):
! CHECK:    %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_8]])  typeparams %[[VAL_2]] : (!fir.ref<!fir.array<20x!fir.char<1,10>>>, index, index) -> !fir.ref<!fir.char<1,10>>
! CHECK:    %[[VAL_10:.*]] = hlfir.as_expr %[[VAL_9]] : (!fir.ref<!fir.char<1,10>>) -> !hlfir.expr<!fir.char<1,10>>
! CHECK:    hlfir.yield_element %[[VAL_10]] : !hlfir.expr<!fir.char<1,10>>
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_7]]

subroutine chained_elemental(x, y, z)
  integer :: x(100), y(100), z(100)
  x = x+y+z
end subroutine
! CHECK-LABEL: func.func @_QPchained_elemental(
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_4:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_8:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_7:[^)]*]]) {{.*}}y
! CHECK:  %[[VAL_11:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_10:[^)]*]]) {{.*}}z
! CHECK:  %[[VAL_12:.*]] = hlfir.elemental %[[VAL_4]] : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_13:.*]]: index):
! CHECK:    %[[VAL_14:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_15:.*]] = hlfir.designate %[[VAL_8]]#0 (%[[VAL_13]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_16:.*]] = fir.load %[[VAL_14]] : !fir.ref<i32>
! CHECK:    %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:    %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_17]] : i32
! CHECK:    hlfir.yield_element %[[VAL_18]] : i32
! CHECK:  }
! CHECK:  %[[VAL_19:.*]] = hlfir.elemental %[[VAL_4]] : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_20:.*]]: index):
! CHECK:    %[[VAL_21:.*]] = hlfir.apply %[[VAL_22:.*]], %[[VAL_20]] : (!hlfir.expr<100xi32>, index) -> i32
! CHECK:    %[[VAL_23:.*]] = hlfir.designate %[[VAL_11]]#0 (%[[VAL_20]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_24:.*]] = fir.load %[[VAL_23]] : !fir.ref<i32>
! CHECK:    %[[VAL_25:.*]] = arith.addi %[[VAL_21]], %[[VAL_24]] : i32
! CHECK:    hlfir.yield_element %[[VAL_25]] : i32
! CHECK:  }
! CHECK: hlfir.assign
! CHECK: hlfir.destroy %[[VAL_19]]
! CHECK: hlfir.destroy %[[VAL_12]]

subroutine lower_bounds(x)
  integer :: x(2:101)
  call test((x))
end subroutine
! CHECK-LABEL: func.func @_QPlower_bounds(
! CHECK:  %[[VAL_1:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 100 : index
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare %{{.*}}(%[[VAL_3:[^)]*]]) {{.*}}x
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_6:.*]] = hlfir.elemental %[[VAL_5]] : (!fir.shape<1>) -> !hlfir.expr<100xi32> {
! CHECK:  ^bb0(%[[VAL_7:.*]]: index):
! CHECK:    %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_9:.*]] = arith.subi %[[VAL_1]], %[[VAL_8]] : index
! CHECK:    %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_9]] : index
! CHECK:    %[[VAL_11:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_10]])  : (!fir.box<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:    %[[VAL_12:.*]] = fir.load %[[VAL_11]] : !fir.ref<i32>
! CHECK:    %[[VAL_13:.*]] = hlfir.no_reassoc %[[VAL_12]] : i32
! CHECK:    hlfir.yield_element %[[VAL_13]] : i32
! CHECK:  }
! CHECK: fir.call
! CHECK: hlfir.destroy %[[VAL_6]]
