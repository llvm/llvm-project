! RUN: bbc -emit-hlfir %s -o - | FileCheck %s
!
! Test for folding of nested integer kind conversions of a
! non-constant operand, e.g. INT(INT(x,KIND=4),KIND=2).

subroutine double_convert_kind_mismatch(x)
  integer(8), intent(in) :: x
  print *, int(int(x, kind=4), kind=2)
end subroutine
! CHECK-LABEL: func.func @_QPdouble_convert_kind_mismatch(
! CHECK: %[[X:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK: %[[T1:.*]] = fir.convert %[[X]] : (i64) -> i32
! CHECK: %[[T2:.*]] = fir.convert %[[T1]] : (i32) -> i16
! CHECK: fir.call @_FortranAioOutputInteger16(%{{.*}}, %[[T2]])
