! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_generic() {
subroutine distribute_parallel_do_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  call bar() !< Prevents this from being SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  call bar() !< Prevents this from being SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do

  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end target teams
end subroutine distribute_parallel_do_generic

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_spmd() {
subroutine distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do
  !$omp end target teams
end subroutine distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_generic() {
subroutine distribute_parallel_do_simd_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  call bar() !< Prevents this from being SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  call bar() !< Prevents this from being SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd

  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams
end subroutine distribute_parallel_do_simd_generic

! CHECK-LABEL: func.func @_QPdistribute_parallel_do_simd_spmd() {
subroutine distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams
end subroutine distribute_parallel_do_simd_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_parallel_do_spmd() {
subroutine teams_distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute parallel do
  !$omp end target
end subroutine teams_distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_parallel_do_simd_spmd() {
subroutine teams_distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute parallel do simd
  !$omp end target
end subroutine teams_distribute_parallel_do_simd_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_parallel_do_spmd() {
subroutine target_teams_distribute_parallel_do_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute parallel do
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute parallel do
end subroutine target_teams_distribute_parallel_do_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_parallel_do_simd_spmd() {
subroutine target_teams_distribute_parallel_do_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute parallel do simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute parallel do simd
end subroutine target_teams_distribute_parallel_do_simd_spmd
