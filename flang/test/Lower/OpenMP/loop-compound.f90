! This test checks lowering of OpenMP compound (combined and composite) loop
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO When composite constructs are supported add:
  ! - DISTRIBUTE PARALLEL DO SIMD
  ! - DISTRIBUTE PARALLEL DO
  ! - DISTRIBUTE SIMD
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! - TARGET TEAMS DISTRIBUTE SIMD
  ! - TASKLOOP SIMD
  ! - TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TEAMS DISTRIBUTE PARALLEL DO
  ! - TEAMS DISTRIBUTE SIMD

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do
  do i = 1, 10
  end do
  !$omp end parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd
  do i = 1, 10
  end do
  !$omp end target simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute
  do i = 1, 10
  end do
  !$omp end teams distribute
end program main
