! This test checks lowering of OpenMP compound (combined and composite) loop
! constructs.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO When composite constructs are supported add:
  ! - TASKLOOP SIMD

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute simd
  do i = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.wsloop
  ! CHECK-NEXT: omp.simd
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
  ! CHECK-NEXT: omp.simd
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
  ! CHECK-NEXT: omp.simd
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
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

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

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK-NEXT: omp.simd
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd
  do i = 1, 10
  end do
  !$omp end teams distribute simd
end program main
