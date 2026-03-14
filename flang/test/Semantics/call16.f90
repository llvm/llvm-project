! RUN: %python %S/test_errors.py %s %flang_fc1

! Test that intrinsic functions used as subroutines and vice versa are caught.

subroutine test(x, t)
 intrinsic :: sin, cpu_time
 !ERROR: Cannot call function 'sin' like a subroutine
 call sin(x)
 !ERROR: Cannot call subroutine 'cpu_time' like a function
 x = cpu_time(t)
end subroutine


