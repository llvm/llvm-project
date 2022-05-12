! RUN: %python %S/test_errors.py %s %flang_fc1
! Enforce limits on rank + corank
module m
  !ERROR: 'x' has rank 16, which is greater than the maximum supported rank 15
  real :: x(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  !ERROR: 'y' has rank 16, which is greater than the maximum supported rank 15
  real, allocatable :: y(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  !ERROR: 'z' has rank 16, which is greater than the maximum supported rank 15
  real, pointer :: z(:,:,:,:,:,:,:,:,:,:,:,:,:,:,:,:)
  !ERROR: 'w' has rank 16, which is greater than the maximum supported rank 15
  real, dimension(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1) :: w
  !ERROR: 'a' has rank 15 and corank 1, whose sum is greater than the maximum supported rank 15
  real :: a(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)[*]
  !ERROR: 'b' has rank 14 and corank 2, whose sum is greater than the maximum supported rank 15
  real :: b(1,1,1,1,1,1,1,1,1,1,1,1,1,1)[1,*]
  !ERROR: 'c' has rank 14 and corank 2, whose sum is greater than the maximum supported rank 15
  real :: c
  dimension :: c(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  codimension :: c[1,*]
  interface
    !ERROR: 'foo' has rank 16, which is greater than the maximum supported rank 15
    real function foo()
      dimension :: foo(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
    end function
  end interface
 contains
  function bar() result(res)
    !ERROR: 'res' has rank 16, which is greater than the maximum supported rank 15
    real :: res(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
  end function
end module
