! This test checks lowering of OpenMP combined loop constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO When DISTRIBUTE, TASKLOOP and TEAMS are supported add:
  ! - DISTRIBUTE PARALLEL DO SIMD
  ! - DISTRIBUTE PARALLEL DO
  ! - DISTRIBUTE SIMD
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! - TARGET TEAMS DISTRIBUTE SIMD
  ! - TARGET TEAMS DISTRIBUTE
  ! - TASKLOOP SIMD
  ! - TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TEAMS DISTRIBUTE PARALLEL DO
  ! - TEAMS DISTRIBUTE SIMD
  ! - TEAMS DISTRIBUTE

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.simdloop
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.parallel
  ! CHECK: omp.simdloop
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
  ! CHECK: omp.simdloop
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
  ! CHECK: omp.simdloop
  !$omp target simd
  do i = 1, 10
  end do
  !$omp end target simd
end program main
