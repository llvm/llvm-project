! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell

! Test that intrinsic functions used as subroutines and vice versa are caught.

subroutine test(x, t)
 intrinsic :: sin, cpu_time
 !ERROR: Cannot use intrinsic function 'sin' as a subroutine
 call sin(x)
 !ERROR: Cannot use intrinsic subroutine 'cpu_time' as a function
 x = cpu_time(t)
end subroutine


