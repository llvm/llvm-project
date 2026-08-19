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
! CHECK-DAG: acc.routine @[[FOO_N_ROUTINE:.*]] func(@_QPfoo_n) vector ([#acc.device_type<nvidia>]){{$}}
! CHECK-DAG: acc.routine @[[FOO_M_ROUTINE:.*]] func(@_QPfoo_m) worker ([#acc.device_type<multicore>]){{$}}
! CHECK-DAG: func.func private @_QPfoo_n({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[FOO_N_ROUTINE]]]>}
! CHECK-DAG: func.func private @_QPfoo_m({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[FOO_M_ROUTINE]]]>}

subroutine s_bind_devtype_merged_target(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(foo_merge) device_type(nvidia) vector bind(foo_dev) device_type(multicore) worker bind(foo_dev)
  external :: foo_merge
  !$acc parallel loop
  do i = 1, n
    call foo_merge(x(i))
  end do
end subroutine

! CHECK-DAG: acc.routine @{{.*}} func(@_QPfoo_merge) bind(@_QPfoo_dev [#acc.device_type<nvidia>], @_QPfoo_dev [#acc.device_type<multicore>]) worker ([#acc.device_type<multicore>]) vector ([#acc.device_type<nvidia>])
! CHECK-DAG: acc.routine @[[FOO_DEV_ROUTINE:.*]] func(@_QPfoo_dev) worker ([#acc.device_type<multicore>]) vector ([#acc.device_type<nvidia>]){{$}}
! CHECK-DAG: func.func private @_QPfoo_dev({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[FOO_DEV_ROUTINE]]]>}

subroutine s_bind_before_modality(n, x)
  integer :: n, i
  real :: x(n)
  !$acc routine(bar) device_type(nvidia) bind(bar_n) vector device_type(multicore) bind(bar_m) seq
  external :: bar
  !$acc parallel loop
  do i = 1, n
    call bar(x(i))
  end do
end subroutine

! CHECK-DAG: acc.routine @{{.*}} func(@_QPbar) bind(@_QPbar_n [#acc.device_type<nvidia>], @_QPbar_m [#acc.device_type<multicore>]) vector ([#acc.device_type<nvidia>]) seq ([#acc.device_type<multicore>])
! CHECK-DAG: acc.routine @[[BAR_N_ROUTINE:.*]] func(@_QPbar_n) vector ([#acc.device_type<nvidia>]){{$}}
! CHECK-DAG: acc.routine @[[BAR_M_ROUTINE:.*]] func(@_QPbar_m) seq ([#acc.device_type<multicore>]){{$}}
! CHECK-DAG: func.func private @_QPbar_n({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[BAR_N_ROUTINE]]]>}
! CHECK-DAG: func.func private @_QPbar_m({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@[[BAR_M_ROUTINE]]]>}
