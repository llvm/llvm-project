!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror
!WARNING: Value of local variable 'x' is never used [-Wunused-variable]
real, allocatable:: x(:)
allocate(x(1))
print *, lbound(x)
end
