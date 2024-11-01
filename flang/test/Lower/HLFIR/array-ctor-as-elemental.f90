! Test lowering of array constructors as hlfir.elemental.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

subroutine test_as_simple_elemental(n)
  integer :: n
  call takes_int([(n+i, i=1,4)])
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_simple_elemental(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare {{.*}}En
! CHECK:  %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_9:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<4xi32> {
! CHECK:  ^bb0(%[[VAL_10:.*]]: index):
! CHECK:    %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : index
! CHECK:    %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_7]] : index
! CHECK:    %[[VAL_13:.*]] = arith.addi %[[VAL_5]], %[[VAL_12]] : index
! CHECK:    %[[VAL_14:.*]] = fir.load %[[VAL_1]]#0 : !fir.ref<i32>
! CHECK:    %[[VAL_15:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:    %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : i32
! CHECK:    hlfir.yield_element %[[VAL_16]] : i32
! CHECK:  }
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_9]] : !hlfir.expr<4xi32>

subroutine test_as_strided_elemental(lb, ub, stride)
  integer(8) :: lb, ub, stride
  call takes_int([(i, i=lb,ub,stride)])
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_strided_elemental(
! CHECK:  %[[VAL_3:.*]]:2 = hlfir.declare {{.*}}Elb
! CHECK:  %[[VAL_4:.*]]:2 = hlfir.declare {{.*}}Estride
! CHECK:  %[[VAL_5:.*]]:2 = hlfir.declare {{.*}}Eub
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_8:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_9:.*]] = arith.subi %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:  %[[VAL_10:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:  %[[VAL_12:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_13:.*]] = arith.divsi %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:  %[[VAL_14:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:  %[[VAL_16:.*]] = arith.select %[[VAL_15]], %[[VAL_13]], %[[VAL_14]] : i64
! CHECK:  %[[VAL_17:.*]] = arith.addi %[[VAL_6]], %[[VAL_16]] : i64
! CHECK:  %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
! CHECK:  %[[VAL_19:.*]] = fir.shape %[[VAL_18]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_20:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i64) -> index
! CHECK:  %[[VAL_22:.*]] = fir.load %[[VAL_4]]#0 : !fir.ref<i64>
! CHECK:  %[[VAL_23:.*]] = fir.convert %[[VAL_22]] : (i64) -> index
! CHECK:  %[[VAL_24:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_25:.*]] = hlfir.elemental %[[VAL_19]] : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:  ^bb0(%[[VAL_26:.*]]: index):
! CHECK:    %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_24]] : index
! CHECK:    %[[VAL_28:.*]] = arith.muli %[[VAL_27]], %[[VAL_23]] : index
! CHECK:    %[[VAL_29:.*]] = arith.addi %[[VAL_21]], %[[VAL_28]] : index
! CHECK:    %[[VAL_30:.*]] = fir.convert %[[VAL_29]] : (index) -> i32
! CHECK:    hlfir.yield_element %[[VAL_30]] : i32
! CHECK:  }
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_25]] : !hlfir.expr<?xi32>

subroutine test_as_elemental_with_pure_call(n)
  interface
    integer pure function foo(i)
      integer, value :: i
    end function
  end interface
  integer :: n
  call takes_int([(foo(i), i=1,4)])
end subroutine
! CHECK-LABEL: func.func @_QPtest_as_elemental_with_pure_call(
! CHECK-SAME:                                                 %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFtest_as_elemental_with_pure_callEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:  %[[VAL_2:.*]] = arith.constant 4 : index
! CHECK:  %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_4:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : i64
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_9:.*]] = hlfir.elemental %[[VAL_3]] : (!fir.shape<1>) -> !hlfir.expr<4xi32> {
! CHECK:  ^bb0(%[[VAL_10:.*]]: index):
! CHECK:    %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : index
! CHECK:    %[[VAL_12:.*]] = arith.muli %[[VAL_11]], %[[VAL_7]] : index
! CHECK:    %[[VAL_13:.*]] = arith.addi %[[VAL_5]], %[[VAL_12]] : index
! CHECK:    %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (index) -> i32
! CHECK:    %[[VAL_15:.*]] = fir.call @_QPfoo(%[[VAL_14]]) fastmath<contract> : (i32) -> i32
! CHECK:    hlfir.yield_element %[[VAL_15]] : i32
! CHECK:  }
! CHECK:  fir.call
! CHECK:  hlfir.destroy %[[VAL_9]] : !hlfir.expr<4xi32>

! CHECK-LABEL: func.func @_QPtest_with_impure_call(
subroutine test_with_impure_call(n)
  interface
    integer function impure_foo(i)
      integer, value :: i
    end function
  end interface
  integer :: n
  call takes_int([(impure_foo(i), i=1,4)])
end subroutine
! CHECK-NOT: hlfir.elemental
! CHECK:  return
