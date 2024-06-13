! This test checks lowering of OpenMP combined loop constructs.

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
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  !$omp parallel do simd
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
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
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.simd
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
  !$omp target teams distribute
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------

  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  !$omp teams distribute
  do i = 1, 10
  end do
  !$omp end teams distribute
end program main
