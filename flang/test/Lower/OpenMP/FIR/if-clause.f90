! This test checks lowering of OpenMP IF clauses.

! RUN: bbc -fopenmp -emit-fir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-fir %s -o - | FileCheck %s

program main
  integer :: i

  ! TODO When they are supported, add tests for:
  ! - DISTRIBUTE PARALLEL DO
  ! - DISTRIBUTE PARALLEL DO SIMD
  ! - DISTRIBUTE SIMD
  ! - PARALLEL SECTIONS
  ! - PARALLEL WORKSHARE
  ! - TARGET PARALLEL
  ! - TARGET TEAMS DISTRIBUTE
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! - TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TARGET TEAMS DISTRIBUTE SIMD
  ! - TARGET UPDATE
  ! - TASKLOOP
  ! - TASKLOOP SIMD
  ! - TEAMS DISTRIBUTE
  ! - TEAMS DISTRIBUTE PARALLEL DO
  ! - TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! - TEAMS DISTRIBUTE SIMD

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp do simd
  do i = 1, 10
  end do
  !$omp end do simd

  ! CHECK:      omp.wsloop
  !$omp do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end do simd

  ! CHECK:      omp.wsloop
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
  !$omp parallel do
  do i = 1, 10
  end do
  !$omp end parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
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
  !$omp parallel do simd
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  !$omp parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd
  
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  !$omp parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end parallel do simd
  
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp parallel do simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd
  
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.wsloop
  !$omp parallel do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.simdloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp simd
  do i = 1, 10
  end do
  !$omp end simd

  ! CHECK:      omp.simdloop
  ! CHECK-SAME: if({{.*}})
  !$omp simd if(.true.)
  do i = 1, 10
  end do
  !$omp end simd

  ! CHECK:      omp.simdloop
  ! CHECK-SAME: if({{.*}})
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
  ! CHECK:      omp.target.data
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target data map(tofrom: i)
  !$omp end target data

  ! CHECK:      omp.target.data
  ! CHECK-SAME: if({{.*}})
  !$omp target data map(tofrom: i) if(.true.)
  !$omp end target data

  ! CHECK:      omp.target.data
  ! CHECK-SAME: if({{.*}})
  !$omp target data map(tofrom: i) if(target data: .true.)
  !$omp end target data

  ! ----------------------------------------------------------------------------
  ! TARGET ENTER DATA
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target.enterdata
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: map
  !$omp target enter data map(to: i)

  ! CHECK:      omp.target.enterdata
  ! CHECK-SAME: if({{.*}})
  !$omp target enter data map(to: i) if(.true.)

  ! CHECK:      omp.target.enterdata
  ! CHECK-SAME: if({{.*}})
  !$omp target enter data map(to: i) if(target enter data: .true.)

  ! ----------------------------------------------------------------------------
  ! TARGET EXIT DATA
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target.exitdata
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: map
  !$omp target exit data map(from: i)

  ! CHECK:      omp.target.exitdata
  ! CHECK-SAME: if({{.*}})
  !$omp target exit data map(from: i) if(.true.)

  ! CHECK:      omp.target.exitdata
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
  !$omp target parallel do
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp target parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  !$omp target parallel do if(target: .true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
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
  !$omp target parallel do simd
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
  !$omp target parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.parallel
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.wsloop
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
  !$omp target parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.simdloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target simd
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simdloop
  ! CHECK-SAME: if({{.*}})
  !$omp target simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simdloop
  ! CHECK-SAME: if({{.*}})
  !$omp target simd if(target: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-SAME: if({{.*}})
  ! CHECK:      omp.simdloop
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  !$omp target simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! CHECK:      omp.target
  ! CHECK-NOT:  if({{.*}})
  ! CHECK-SAME: {
  ! CHECK:      omp.simdloop
  ! CHECK-SAME: if({{.*}})
  !$omp target simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target simd

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
