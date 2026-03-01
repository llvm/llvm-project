! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! Test that LEN intrinsic in elemental function result specification expression
! is lowered efficiently in HLFIR, especially when it involves expressions
! like concatenation that can be folded or handled without materialization.

elemental function add_suffix(str) result(res)
  character(len=*), intent(in) :: str
  character(len=len(str // "abc")) :: res
  res = str // "xyz"
end function

! CHECK-LABEL: func.func @_QPadd_suffix(
! CHECK:         hlfir.declare {{.*}} "_QFadd_suffixEstr"
! CHECK-NOT:     hlfir.concat
! CHECK:         hlfir.declare {{.*}} "_QFadd_suffixEres"
! CHECK:         hlfir.concat
! CHECK:         hlfir.assign

subroutine test_call(s, r)
  character(*), intent(in) :: s(:)
  character(*), intent(out) :: r(:)
  interface
    elemental function add_suffix(str) result(res)
      character(len=*), intent(in) :: str
      character(len=len(str // "abc")) :: res
    end function
  end interface
  r = add_suffix(s)
end subroutine

! CHECK-LABEL: func.func @_QPtest_call(
! CHECK:         hlfir.declare {{.*}} "_QFtest_callEs"
! CHECK-NOT:     hlfir.concat
! CHECK:         hlfir.elemental {{.*}} {
! CHECK:           hlfir.designate
! CHECK:           fir.alloca
! CHECK:           hlfir.declare
! CHECK:           fir.call @_QPadd_suffix
! CHECK:         }

subroutine test_trim(s, n)
  character(*) :: s
  integer :: n
  n = len(trim(s))
end subroutine

! CHECK-LABEL: func.func @_QPtest_trim(
! CHECK:         hlfir.declare {{.*}} "_QFtest_trimEs"
! CHECK:         %[[TRIM:.*]] = hlfir.char_trim
! CHECK-NEXT:    %[[LEN:.*]] = hlfir.get_length %[[TRIM]]
! CHECK-NEXT:    %[[LEN_I32:.*]] = fir.convert %[[LEN]] : (index) -> i32
! CHECK-NEXT:    hlfir.assign %[[LEN_I32]] to {{.*}}
! CHECK-NEXT:    hlfir.destroy %[[TRIM]]
