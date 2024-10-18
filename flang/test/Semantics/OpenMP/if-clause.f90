! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! Check OpenMP 'if' clause validity for all directives that can have it

program main
  integer :: i

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp teams
  !$omp distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !$omp distribute parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp distribute parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do

  !ERROR: At most one IF clause can appear on the DISTRIBUTE PARALLEL DO directive
  !$omp distribute parallel do if(.true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do
  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams
  !$omp distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  !$omp distribute parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp distribute parallel do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end distribute parallel do simd
  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams
  !$omp distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end distribute simd

  !$omp distribute simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end distribute simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp distribute simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end distribute simd

  !ERROR: At most one IF clause can appear on the DISTRIBUTE SIMD directive
  !$omp distribute simd if(.true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end distribute simd
  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end do simd

  !$omp do simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end do simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end do simd

  !ERROR: At most one IF clause can appear on the DO SIMD directive
  !$omp do simd if(.true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL
  ! ----------------------------------------------------------------------------
  !$omp parallel if(.true.)
  !$omp end parallel

  !$omp parallel if(parallel: .true.)
  !$omp end parallel

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel if(target: .true.)
  !$omp end parallel

  !ERROR: At most one IF clause can appear on the PARALLEL directive
  !$omp parallel if(.true.) if(parallel: .false.)
  !$omp end parallel

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  !$omp parallel do if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do

  !ERROR: At most one IF clause can appear on the PARALLEL DO directive
  !$omp parallel do if(.true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end parallel do

  ! ----------------------------------------------------------------------------
  ! PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  !$omp parallel do simd if(parallel: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! ----------------------------------------------------------------------------
  ! PARALLEL SECTIONS
  ! ----------------------------------------------------------------------------
  !$omp parallel sections if(.true.)
  !$omp end parallel sections

  !$omp parallel sections if(parallel: .true.)
  !$omp end parallel sections

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel sections if(target: .true.)
  !$omp end parallel sections

  !ERROR: At most one IF clause can appear on the PARALLEL SECTIONS directive
  !$omp parallel sections if(.true.) if(parallel: .false.)
  !$omp end parallel sections

  ! ----------------------------------------------------------------------------
  ! PARALLEL WORKSHARE
  ! ----------------------------------------------------------------------------
  !$omp parallel workshare if(.true.)
  !$omp end parallel workshare

  !$omp parallel workshare if(parallel: .true.)
  !$omp end parallel workshare

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp parallel workshare if(target: .true.)
  !$omp end parallel workshare

  !ERROR: At most one IF clause can appear on the PARALLEL WORKSHARE directive
  !$omp parallel workshare if(.true.) if(parallel: .false.)
  !$omp end parallel workshare

  ! ----------------------------------------------------------------------------
  ! SIMD
  ! ----------------------------------------------------------------------------
  !$omp simd if(.true.)
  do i = 1, 10
  end do
  !$omp end simd

  !$omp simd if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end simd

  !ERROR: At most one IF clause can appear on the SIMD directive
  !$omp simd if(.true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end simd

  ! ----------------------------------------------------------------------------
  ! TARGET
  ! ----------------------------------------------------------------------------
  !$omp target if(.true.)
  !$omp end target

  !$omp target if(target: .true.)
  !$omp end target

  !ERROR: Unmatched directive name modifier PARALLEL on the IF clause
  !$omp target if(parallel: .true.)
  !$omp end target

  !ERROR: At most one IF clause can appear on the TARGET directive
  !$omp target if(.true.) if(target: .false.)
  !$omp end target

  ! ----------------------------------------------------------------------------
  ! TARGET DATA
  ! ----------------------------------------------------------------------------
  !$omp target data map(tofrom: i) if(.true.)
  !$omp end target data

  !$omp target data map(tofrom: i) if(target data: .true.)
  !$omp end target data

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp target data map(tofrom: i) if(target: .true.)
  !$omp end target data

  !ERROR: At most one IF clause can appear on the TARGET DATA directive
  !$omp target data map(tofrom: i) if(.true.) if(target data: .false.)
  !$omp end target data

  ! ----------------------------------------------------------------------------
  ! TARGET ENTER DATA
  ! ----------------------------------------------------------------------------
  !$omp target enter data map(to: i) if(.true.)

  !$omp target enter data map(to: i) if(target enter data: .true.)

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp target enter data map(to: i) if(target: .true.)

  !ERROR: At most one IF clause can appear on the TARGET ENTER DATA directive
  !$omp target enter data map(to: i) if(.true.) if(target enter data: .false.)

  ! ----------------------------------------------------------------------------
  ! TARGET EXIT DATA
  ! ----------------------------------------------------------------------------
  !$omp target exit data map(from: i) if(.true.)

  !$omp target exit data map(from: i) if(target exit data: .true.)

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp target exit data map(from: i) if(target: .true.)
  
  !ERROR: At most one IF clause can appear on the TARGET EXIT DATA directive
  !$omp target exit data map(from: i) if(.true.) if(target exit data: .false.)

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL
  ! ----------------------------------------------------------------------------
  !$omp target parallel if(.true.)
  !$omp end target parallel

  !$omp target parallel if(target: .true.) if(parallel: .false.)
  !$omp end target parallel

  !ERROR: Unmatched directive name modifier SIMD on the IF clause
  !$omp target parallel if(simd: .true.)
  !$omp end target parallel

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp target parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  !$omp target parallel do if(target: .true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  !ERROR: Unmatched directive name modifier SIMD on the IF clause
  !$omp target parallel do if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp target parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  !$omp target parallel do simd if(target: .true.) if(parallel: .false.) &
  !$omp&                        if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  !ERROR: Unmatched directive name modifier TEAMS on the IF clause
  !$omp target parallel do simd if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end target parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET SIMD
  ! ----------------------------------------------------------------------------
  !$omp target simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target simd

  !$omp target simd if(target: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target simd

  !ERROR: Unmatched directive name modifier PARALLEL on the IF clause
  !$omp target simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS
  ! ----------------------------------------------------------------------------
  !$omp target teams if(.true.)
  !$omp end target teams

  !$omp target teams if(target: .true.) if(teams: .false.)
  !$omp end target teams

  !ERROR: Unmatched directive name modifier PARALLEL on the IF clause
  !$omp target teams if(parallel: .true.)
  !$omp end target teams

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------
  !$omp target teams distribute if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  !$omp target teams distribute if(target: .true.) if(teams: .false.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  !ERROR: Unmatched directive name modifier PARALLEL on the IF clause
  !$omp target teams distribute if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp target teams distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do &
  !$omp&   if(target: .true.) if(teams: .false.) if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  !ERROR: Unmatched directive name modifier SIMD on the IF clause
  !$omp target teams distribute parallel do if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp target teams distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd &
  !$omp&   if(target: .true.) if(teams: .false.) if(parallel: .true.) &
  !$omp&   if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  !ERROR: Unmatched directive name modifier TASK on the IF clause
  !$omp target teams distribute parallel do simd if(task: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TARGET TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp target teams distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  !$omp target teams distribute simd &
  !$omp&   if(target: .true.) if(teams: .false.) if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  !ERROR: Unmatched directive name modifier PARALLEL on the IF clause
  !$omp target teams distribute simd if(parallel: .true.)
  do i = 1, 10
  end do
  !$omp end target teams distribute simd

  ! ----------------------------------------------------------------------------
  ! TARGET UPDATE
  ! ----------------------------------------------------------------------------
  !$omp target update to(i) if(.true.)
  
  !$omp target update to(i) if(target update: .true.)

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp target update to(i) if(target: .true.)

  !ERROR: At most one IF clause can appear on the TARGET UPDATE directive
  !$omp target update to(i) if(.true.) if(target update: .false.)

  ! ----------------------------------------------------------------------------
  ! TASK
  ! ----------------------------------------------------------------------------
  !$omp task if(.true.)
  !$omp end task

  !$omp task if(task: .true.)
  !$omp end task

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp task if(target: .true.)
  !$omp end task

  !ERROR: At most one IF clause can appear on the TASK directive
  !$omp task if(.true.) if(task: .false.)
  !$omp end task

  ! ----------------------------------------------------------------------------
  ! TASKLOOP
  ! ----------------------------------------------------------------------------
  !$omp taskloop if(.true.)
  do i = 1, 10
  end do
  !$omp end taskloop

  !$omp taskloop if(taskloop: .true.)
  do i = 1, 10
  end do
  !$omp end taskloop

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp taskloop if(target: .true.)
  do i = 1, 10
  end do
  !$omp end taskloop

  !ERROR: At most one IF clause can appear on the TASKLOOP directive
  !$omp taskloop if(.true.) if(taskloop: .false.)
  do i = 1, 10
  end do
  !$omp end taskloop

  ! ----------------------------------------------------------------------------
  ! TASKLOOP SIMD
  ! ----------------------------------------------------------------------------
  !$omp taskloop simd if(.true.)
  do i = 1, 10
  end do
  !$omp end taskloop simd

  !$omp taskloop simd if(taskloop: .true.) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end taskloop simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp taskloop simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end taskloop simd

  ! ----------------------------------------------------------------------------
  ! TEAMS
  ! ----------------------------------------------------------------------------
  !$omp teams if(.true.)
  !$omp end teams

  !$omp teams if(teams: .true.)
  !$omp end teams

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp teams if(target: .true.)
  !$omp end teams

  !ERROR: At most one IF clause can appear on the TEAMS directive
  !$omp teams if(.true.) if(teams: .false.)
  !$omp end teams

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE
  ! ----------------------------------------------------------------------------
  !$omp teams distribute if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  !$omp teams distribute if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp teams distribute if(target: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  !ERROR: At most one IF clause can appear on the TEAMS DISTRIBUTE directive
  !$omp teams distribute if(.true.) if(teams: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO
  ! ----------------------------------------------------------------------------
  !$omp teams distribute parallel do if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  !$omp teams distribute parallel do if(teams: .true.) if(parallel: .false.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp teams distribute parallel do if(target: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE PARALLEL DO SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams distribute parallel do simd if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  !$omp teams distribute parallel do simd &
  !$omp&   if(teams: .true.) if(parallel: .true.) if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp teams distribute parallel do simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute parallel do simd

  ! ----------------------------------------------------------------------------
  ! TEAMS DISTRIBUTE SIMD
  ! ----------------------------------------------------------------------------
  !$omp teams distribute simd if(.true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  !$omp teams distribute simd if(teams: .true.) if(simd: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd

  !ERROR: Unmatched directive name modifier TARGET on the IF clause
  !$omp teams distribute simd if(target: .true.)
  do i = 1, 10
  end do
  !$omp end teams distribute simd
end program main
