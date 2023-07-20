! RUN: %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

program test
  vector(integer(4)) :: arg1, arg2, r
  vector(real(4)) :: rr
  integer :: i

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sld(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0-15
  r = vec_sld(arg1, arg2, 17)

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sldw(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0-3
  r = vec_sldw(arg1, arg2, 5)

!ERROR: Actual argument #2 must be a constant expression
  rr = vec_ctf(arg1, i)
! ERROR: Argument #2 must be a constant expression in range 0-31
  rr = vec_ctf(arg1, 37)
end program test
