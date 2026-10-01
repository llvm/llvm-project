! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module target_composite_private_edge_types
  type :: dt_ptr
    integer, pointer :: p
  end type
end module

subroutine target_parallel_do_private_if(n, out)
  integer :: n, out, i
  out = 0
  !$omp target parallel do private(n) if(n > 0) map(tofrom: out)
  do i = 1, 4
    n = i
    out = out + n
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_if
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.map.info {{.*}} name("n")
! CHECK: omp.map.info {{.*}} name("i")
! CHECK: omp.target

subroutine target_parallel_do_private_schedule(n, out)
  integer :: n, out, i
  out = 0
  !$omp target parallel do private(n) schedule(static, n) map(tofrom: out)
  do i = 1, 4
    n = i
    out = out + n
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_schedule
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.map.info {{.*}} name("n")
! CHECK: omp.map.info {{.*}} name("i")
! CHECK: omp.target

subroutine target_teams_distribute_parallel_do_private(s, n, out)
  integer :: s, n, out, i
  out = 0
  !$omp target teams distribute parallel do private(s) map(tofrom: out)
  do i = 1, n
    s = i
    out = out + s
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_parallel_do_private
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.map.info {{.*}} name("i")
! CHECK: omp.map.info {{.*}} name("n")
! CHECK: omp.target

subroutine target_parallel_do_private_derived_pointer()
  use target_composite_private_edge_types
  type(dt_ptr) :: d
  integer, target :: x
  integer :: i
  d%p => x
  !$omp target parallel do private(d)
  do i = 1, 4
    if (associated(d%p)) d%p = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_derived_pointer
! CHECK: omp.map.info {{.*}} name("d")
! CHECK: omp.target

subroutine target_parallel_do_private_common(out)
  integer :: out, i, c
  common /blk/ c
  out = 0
  !$omp target parallel do private(c) map(tofrom: out)
  do i = 1, 4
    c = i
    out = out + c
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_common
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.map.info {{.*}} name("c")
! CHECK: omp.target

subroutine target_parallel_do_private_array_shape()
  integer :: i, arr(4), out
  out = 0
  !$omp target parallel do private(arr) map(tofrom: out)
  do i = 1, 4
    arr(i) = i
    out = out + arr(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_array_shape
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.target
! CHECK: fir.alloca !fir.array<4xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_parallel_do_private_array_shapeEarr"}
! CHECK: omp.wsloop private(
! CHECK-SAME: @_QFtarget_parallel_do_private_array_shapeEarr_private_4xi32

subroutine target_teams_nested_parallel_private(s, out)
  integer :: s, out
  out = 0
  !$omp target teams map(tofrom: out)
    !$omp parallel private(s)
      s = 1
      out = out + s
    !$omp end parallel
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_nested_parallel_private
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.parallel private(
! CHECK-SAME: @_QFtarget_teams_nested_parallel_privateEs_private_i32

subroutine target_teams_nested_parallel_private_thread_limit(s, out)
  integer :: s, out
  out = 0
  !$omp target teams thread_limit(s) map(tofrom: out)
    !$omp parallel private(s)
      s = 1
      out = out + s
    !$omp end parallel
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_nested_parallel_private_thread_limit
! CHECK: omp.map.info {{.*}} name("out")
! CHECK: omp.map.info {{.*}} name("s")
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.parallel private(
! CHECK-SAME: @_QFtarget_teams_nested_parallel_private_thread_limitEs_private_i32
