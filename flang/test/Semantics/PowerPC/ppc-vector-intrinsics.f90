! RUN: %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

program test
  vector(integer(4)) :: arg1, arg2, r
  vector(real(4)) :: rr
  integer :: i

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sld(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0 to 15
  r = vec_sld(arg1, arg2, 17)

!ERROR: Actual argument #3 must be a constant expression
  r = vec_sldw(arg1, arg2, i)
!ERROR: Argument #3 must be a constant expression in range 0 to 3
  r = vec_sldw(arg1, arg2, 5)

!ERROR: Actual argument #2 must be a constant expression
  rr = vec_ctf(arg1, i)
! ERROR: Argument #2 must be a constant expression in range 0 to 31
  rr = vec_ctf(arg1, 37)
end program test

subroutine test_vec_permi()
  vector(integer(8)) :: arg1, arg2, r
  integer :: arg3
!ERROR: Actual argument #3 must be a constant expression
  r = vec_permi(arg1, arg2, arg3)
! ERROR: Argument #3 must be a constant expression in range 0 to 3
  r = vec_permi(arg1, arg2, 11)
end

subroutine test_vec_splat()
  vector(integer(8)) :: arg1_8, r8
  vector(integer(2)) :: arg1_2, r2
  integer(2) :: arg2
!ERROR: Actual argument #2 must be a constant expression
  r8 = vec_splat(arg1_8, arg2)
!ERROR: Argument #2 must be a constant expression in range 0 to 1
  r8 = vec_splat(arg1_8, 3)
!ERROR: Argument #2 must be a constant expression in range 0 to 7
  r2 = vec_splat(arg1_2, 11)
end

subroutine test_vec_splat_s32()
  integer(4) :: arg1
  vector(integer(4)) :: r
!ERROR: Actual argument #1 must be a constant expression
  r = vec_splat_s32(arg1)
!ERROR: Argument #1 must be a constant expression in range -16 to 15
  r = vec_splat_s32(17)
end
