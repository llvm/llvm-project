! A device_type-specific bind to an undeclared target is also declared.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine s_bind_devtype(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(aclear) seq device_type(nvidia) bind(aclear_dev)
  external :: aclear
  !$acc parallel loop
  do i = 1, n
    call aclear(x(i))
  end do
end subroutine

! CHECK: acc.routine @{{.*}} func(@_QPaclear){{.*}}@_QPaclear_dev
! CHECK: func.func private @_QPaclear_dev
