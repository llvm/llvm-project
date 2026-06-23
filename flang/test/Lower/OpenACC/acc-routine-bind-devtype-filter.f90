! Device_type-specific acc routine bind targets inherit only the clauses for
! the bind target's own device type.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine s_bind_devtype_filter(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(foo) device_type(nvidia) vector bind(foo_n) device_type(multicore) worker bind(foo_m)
  external :: foo
  !$acc parallel loop
  do i = 1, n
    call foo(x(i))
  end do
end subroutine

! CHECK-DAG: acc.routine @{{.*}} func(@_QPfoo) bind(@_QPfoo_n [#acc.device_type<nvidia>], @_QPfoo_m [#acc.device_type<multicore>]) worker ([#acc.device_type<multicore>]) vector ([#acc.device_type<nvidia>])
! CHECK-DAG: acc.routine @{{.*}} func(@_QPfoo_n) vector ([#acc.device_type<nvidia>]){{$}}
! CHECK-DAG: acc.routine @{{.*}} func(@_QPfoo_m) worker ([#acc.device_type<multicore>]){{$}}
! CHECK-DAG: func.func private @_QPfoo_n
! CHECK-DAG: func.func private @_QPfoo_m
