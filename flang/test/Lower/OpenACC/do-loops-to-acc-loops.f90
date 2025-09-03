! This test checks lowering of Fortran do loops and do concurrent loops to OpenACC loop constructs.
! Tests the new functionality that converts Fortran iteration constructs to acc.loop with proper IV handling.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPbasic_do_loop
subroutine basic_do_loop()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop that should be converted to acc.loop
  !$acc kernels
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent
subroutine basic_do_concurrent()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop
  !$acc kernels
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_loop_parallel
subroutine basic_do_loop_parallel()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop with acc parallel that should be converted to acc.loop
  !$acc parallel
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_loop_serial
subroutine basic_do_loop_serial()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop with acc serial that should be converted to acc.loop
  !$acc serial
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end serial

! CHECK: acc.serial {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent_parallel
subroutine basic_do_concurrent_parallel()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop with acc parallel
  !$acc parallel
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent_serial
subroutine basic_do_concurrent_serial()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop with acc serial
  !$acc serial
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end serial

! CHECK: acc.serial {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

end subroutine

! CHECK-LABEL: func.func @_QPmulti_dimension_do_concurrent
subroutine multi_dimension_do_concurrent()
  integer :: i, j, k
  integer, parameter :: n = 10, m = 20, l = 5
  real, dimension(n,m,l) :: a, b

  ! Multi-dimensional do concurrent with multiple iteration variables
  !$acc kernels
  do concurrent (i = 1:n, j = 1:m, k = 1:l)
    a(i,j,k) = b(i,j,k) * 2.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32) = (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32) to (%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32) step (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true, true>}
end subroutine


! CHECK-LABEL: func.func @_QPnested_do_loops
subroutine nested_do_loops()
  integer :: i, j
  integer, parameter :: n = 10, m = 20
  real, dimension(n,m) :: a, b

  ! Nested do loops
  !$acc kernels
  do i = 1, n
    do j = 1, m
      a(i,j) = b(i,j) + i + j
    end do
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

end subroutine

! CHECK-LABEL: func.func @_QPvariable_bounds_and_step
subroutine variable_bounds_and_step(n, start_val, step_val)
  integer, intent(in) :: n, start_val, step_val
  integer :: i
  real, dimension(n) :: a, b

  ! Do loop with variable bounds and step
  !$acc kernels
  do i = start_val, n, step_val
    a(i) = b(i) * 2.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

end subroutine

! CHECK-LABEL: func.func @_QPdifferent_iv_types
subroutine different_iv_types()
  integer(kind=8) :: i8
  integer(kind=4) :: i4
  integer(kind=2) :: i2
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c, d

  ! Test different iteration variable types
  !$acc kernels
  do i8 = 1_8, int(n,8)
    a(i8) = b(i8) + 1.0
  end do
  !$acc end kernels

  !$acc kernels
  do i4 = 1, n
    b(i4) = c(i4) + 1.0
  end do
  !$acc end kernels

  !$acc kernels
  do i2 = 1_2, int(n,2)
    c(i2) = d(i2) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i64) = (%{{.*}} : i64) to (%{{.*}} : i64) step (%{{.*}} : i64)
! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: acc.kernels {
! CHECK: acc.loop {{.*}} control(%{{.*}} : i16) = (%{{.*}} : i16) to (%{{.*}} : i16) step (%{{.*}} : i16)

end subroutine

! -----------------------------------------------------------------------------------------
! Tests for loops that should NOT be converted to acc.loop due to unstructured control flow

! CHECK-LABEL: func.func @_QPinfinite_loop_no_iv
subroutine infinite_loop_no_iv()
  integer :: i
  logical :: condition

  ! Infinite loop with no induction variable - should NOT convert to acc.loop
  !$acc kernels
  do
    i = i + 1
    if (i > 100) exit
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPwhile_like_loop
subroutine while_like_loop()
  integer :: i
  logical :: condition

  i = 1
  condition = .true.

  ! While-like infinite loop - should NOT convert to acc.loop
  !$acc kernels
  do while (condition)
    i = i + 1
    if (i > 100) condition = .false.
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-NOT: acc.loop

end subroutine
