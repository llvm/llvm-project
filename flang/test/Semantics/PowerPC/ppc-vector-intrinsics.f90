! RUN: %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

program test
  vector(integer(4)) :: arg1, arg2, r
  integer :: i

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sld(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0-15
  r = vec_sld(arg1, arg2, 17)

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sldw(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0-3
  r = vec_sldw(arg1, arg2, 5)
end
