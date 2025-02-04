! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPdistribute_generic() {
subroutine distribute_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute
  call bar() !< Prevents this from being Generic-SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute
  call bar() !< Prevents this from being Generic-SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute

  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute
  !$omp end target teams
end subroutine distribute_generic

! CHECK-LABEL: func.func @_QPdistribute_spmd() {
subroutine distribute_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute
  !$omp end target teams
end subroutine distribute_spmd

! CHECK-LABEL: func.func @_QPdistribute_simd_generic() {
subroutine distribute_simd_generic()
  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp teams
  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd
  call bar() !< Prevents this from being Generic-SPMD.
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd
  call bar() !< Prevents this from being Generic-SPMD.
  !$omp end target teams

  ! CHECK: omp.target
  ! CHECK-NOT: host_eval({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd

  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd
  !$omp end target teams
end subroutine distribute_simd_generic

! CHECK-LABEL: func.func @_QPdistribute_simd_spmd() {
subroutine distribute_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams
  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd
  !$omp end teams
  !$omp end target

  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams
  !$omp distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end distribute simd
  !$omp end target teams
end subroutine distribute_simd_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_spmd() {
subroutine teams_distribute_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute
  !$omp end target
end subroutine teams_distribute_spmd

! CHECK-LABEL: func.func @_QPteams_distribute_simd_spmd() {
subroutine teams_distribute_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target
  !$omp teams distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end teams distribute simd
  !$omp end target
end subroutine teams_distribute_simd_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_spmd() {
subroutine target_teams_distribute_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute
end subroutine target_teams_distribute_spmd

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_simd_spmd() {
subroutine target_teams_distribute_simd_spmd()
  ! CHECK: omp.target
  ! CHECK-SAME: host_eval({{.*}})
  !$omp target teams distribute simd
  do i = 1, 10
    call foo(i)
  end do
  !$omp end target teams distribute simd
end subroutine target_teams_distribute_simd_spmd
