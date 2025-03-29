! This test checks lowering of OpenMP IF clauses.

! The "if" clause was added to the "simd" directive in OpenMP 5.0, and
! to the "teams" directive in OpenMP 5.2.
! RUN: bbc -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO When they are supported, add tests for:
  ! - PARALLEL SECTIONS
  ! - PARALLEL WORKSHARE
  ! - TARGET UPDATE
  ! - TASKLOOP
  ! - TASKLOOP SIMD

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute parallel do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute simd
  do i = 1, 10
  end do
  !$omp end distribute simd

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute simd

  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp distribute simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end distribute simd

  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end do simd

  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp parallel
  i = 10
  !$omp end parallel

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp parallel if(.true.)
  i = 10
  !$omp end parallel

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp parallel if(parallel: .true.)
  i = 10
  !$omp end parallel

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do
  do i = 1, 10
  end do
  !$omp end parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp parallel do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp simd
  do i = 1, 10
  end do
  !$omp end simd

  ! CHECK:      omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp simd if(.true.)
  do i = 1, 10
  end do
  !$omp end simd

  ! CHECK:      omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end simd

  ! ----------------------------------------------------------------------------
  ! TARGET
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target
  !$omp end target

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  !$omp target if(.true.)
  !$omp end target

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  !$omp target if(target: .true.)
  !$omp end target

  ! ----------------------------------------------------------------------------
  ! TARGET DATA
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target_data
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target data map(tofrom: i)
  !$omp end target data

  ! CHECK:      omp.target_data
  ! CHECK-SAME: if({{.*}})
  !$omp target data map(tofrom: i) if(.true.)
  !$omp end target data

  ! CHECK:      omp.target_data
  ! CHECK-SAME: if({{.*}})
  !$omp target data map(tofrom: i) if(target data: .true.)
  !$omp end target data

  ! ----------------------------------------------------------------------------
  ! TARGET ENTER DATA
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target_enter_data
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: map
  !$omp target enter data map(to: i)

  ! CHECK:      omp.target_enter_data
  ! CHECK-SAME: if({{.*}})
  !$omp target enter data map(to: i) if(.true.)

  ! CHECK:      omp.target_enter_data
  ! CHECK-SAME: if({{.*}})
  !$omp target enter data map(to: i) if(target enter data: .true.)

  ! ----------------------------------------------------------------------------
  ! TARGET EXIT DATA
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target_exit_data
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: map
  !$omp target exit data map(from: i)

  ! CHECK:      omp.target_exit_data
  ! CHECK-SAME: if({{.*}})
  !$omp target exit data map(from: i) if(.true.)

  ! CHECK:      omp.target_exit_data
  ! CHECK-SAME: if({{.*}})
  !$omp target exit data map(from: i) if(target exit data: .true.)

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do if(target: .true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd if(target: .true.) if(parallel: .false.) &
  !$omp&                        if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target parallel
  i = 1
  !$omp end target parallel

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp target parallel if(.true.)
  i = 1
  !$omp end target parallel

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp target parallel if(target: .true.) if(parallel: .false.)
  i = 1
  !$omp end target parallel

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target parallel if(target: .true.)
  i = 1
  !$omp end target parallel

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp target parallel if(parallel: .true.)
  i = 1
  !$omp end target parallel

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd if(target: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute if(target: .true.) if(teams: .false.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do if(target: .true.) if(teams: .false.) if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(target: .true.) if(teams: .false.) if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute parallel do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd if(target: .true.) if(teams: .false.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp target teams distribute simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target teams
  i = 1
  !$omp end target teams

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  !$omp target teams if(.true.)
  i = 1
  !$omp end target teams

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  !$omp target teams if(target: .true.) if(teams: .false.)
  i = 1
  !$omp end target teams

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target teams if(target: .true.)
  i = 1
  !$omp end target teams

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  !$omp target teams if(teams: .true.)
  i = 1
  !$omp end target teams

  ! ----------------------------------------------------------------------------
  ! TASK
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.task
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp task
  !$omp end task

  ! CHECK:      omp.task
  ! CHECK-SAME: if({{.*}})
  !$omp task if(.true.)
  !$omp end task

  ! CHECK:      omp.task
  ! CHECK-SAME: if({{.*}})
  !$omp task if(task: .true.)
  !$omp end task

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute
  do i = 1, 10
  end do
  !$omp end teams distribute

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do if(teams: .true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd if(teams: .false.) if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute parallel do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd if(teams: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.distribute
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK-NEXT: omp.simd
  ! CHECK-SAME: if({{.*}})
  ! CHECK-NEXT: omp.loop_nest
  !$omp teams distribute simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  ! ----------------------------------------------------------------------------
  ! TEAMS
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.teams
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp teams
  i = 1
  !$omp end teams

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  !$omp teams if(.true.)
  i = 1
  !$omp end teams

  ! CHECK:      omp.teams
  ! CHECK-SAME: if({{.*}})
  !$omp teams if(teams: .true.)
  i = 1
  !$omp end teams
end program main
