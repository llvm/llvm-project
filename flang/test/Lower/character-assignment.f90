! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! Simple character assignment tests
subroutine assign1(lhs, rhs)
  character(*, 1) :: lhs, rhs
  lhs = rhs
end subroutine
! CHECK-LABEL:   func.func @_QPassign1(
! CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare{{.*}}"_QFassign1Elhs"
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}"_QFassign1Erhs"
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_4]]#0 : !fir.boxchar<1>, !fir.boxchar<1>
! CHECK:           return
! CHECK:         }

! Test substring assignment
subroutine assign_substring1(str, rhs, lb, ub)
  character(*, 1) :: rhs, str
  integer(8) :: lb, ub
  str(lb:ub) = rhs
end subroutine
! CHECK-LABEL:   func.func @_QPassign_substring1(
! CHECK:           %[[VAL_7:.*]]:2 = hlfir.declare{{.*}}"_QFassign_substring1Erhs"
! CHECK:           %[[VAL_9:.*]]:2 = hlfir.declare{{.*}}"_QFassign_substring1Estr"
! CHECK:           %[[VAL_21:.*]] = hlfir.designate %[[VAL_9]]#0  substr {{.*}}
! CHECK:           hlfir.assign %[[VAL_7]]#0 to %[[VAL_21]] : !fir.boxchar<1>, !fir.boxchar<1>
! CHECK:           return
! CHECK:         }

subroutine assign_constant(lhs)
  character(*, 1) :: lhs
  lhs = "Hello World"
end subroutine
! CHECK-LABEL:   func.func @_QPassign_constant(
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare{{.*}}"_QFassign_constantElhs"
! CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare{{.*}}"_QQclX48656C6C6F20576F726C64"
! CHECK:           hlfir.assign %[[VAL_6]]#0 to %[[VAL_3]]#0 : !fir.ref<!fir.char<1,11>>, !fir.boxchar<1>

subroutine assign_zero_size_array(n)
  character(n), allocatable :: a(:)
  a = [character(n)::]
end subroutine
! CHECK-LABEL:   func.func @_QPassign_zero_size_array(
! CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare{{.*}}"_QFassign_zero_size_arrayEa"
! CHECK:           %[[VAL_24:.*]] = hlfir.as_expr %{{.*}}
! CHECK:           hlfir.assign %[[VAL_24]] to %[[VAL_12]]#0 realloc keep_lhs_len : !hlfir.expr<0x!fir.char<1,?>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
