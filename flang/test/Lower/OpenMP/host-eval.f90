! The "thread_limit" clause was added to the "target" construct in OpenMP 5.1.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s --check-prefixes=BOTH,HOST
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefixes=BOTH,DEVICE

! BOTH-LABEL: func.func @_QPteams
subroutine teams()
  ! BOTH: omp.target

  ! HOST-SAME: host_eval(%{{.*}} -> %[[NUM_TEAMS:.*]], %{{.*}} -> %[[THREAD_LIMIT:.*]] : i32, i32)
  
  ! DEVICE-NOT: host_eval({{.*}})
  ! DEVICE-SAME: {
  !$omp target

  ! BOTH: omp.teams

  ! HOST-SAME: num_teams( to %[[NUM_TEAMS]] : i32) thread_limit(%[[THREAD_LIMIT]] : i32)
  ! DEVICE-SAME: num_teams({{.*}}) thread_limit({{.*}})
  !$omp teams num_teams(1) thread_limit(2)
  call foo()
  !$omp end teams

  !$omp end target

  ! BOTH: omp.teams
  ! BOTH-SAME: num_teams({{.*}}) thread_limit({{.*}}) {
  !$omp teams num_teams(1) thread_limit(2)
  call foo()
  !$omp end teams
end subroutine teams

! BOTH-LABEL: func.func @_QPdistribute_parallel_do
subroutine distribute_parallel_do()
  ! BOTH: omp.target
  
  ! HOST-SAME: host_eval(%{{.*}} -> %[[LB:.*]], %{{.*}} -> %[[UB:.*]], %{{.*}} -> %[[STEP:.*]], %{{.*}} -> %[[NUM_THREADS:.*]] : i32, i32, i32, i32)
  
  ! DEVICE-NOT: host_eval({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.parallel

  ! HOST-SAME: num_threads(%[[NUM_THREADS]] : i32)
  ! DEVICE-SAME: num_threads({{.*}})

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.loop_nest

  ! HOST-SAME: (%{{.*}}) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: host_eval({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo() !< Prevents this from being SPMD.

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  !$omp distribute parallel do num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do
  !$omp end teams
end subroutine distribute_parallel_do

! BOTH-LABEL: func.func @_QPdistribute_parallel_do_simd
subroutine distribute_parallel_do_simd()
  ! BOTH: omp.target
  
  ! HOST-SAME: host_eval(%{{.*}} -> %[[LB:.*]], %{{.*}} -> %[[UB:.*]], %{{.*}} -> %[[STEP:.*]], %{{.*}} -> %[[NUM_THREADS:.*]] : i32, i32, i32, i32)
  
  ! DEVICE-NOT: host_eval({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.parallel

  ! HOST-SAME: num_threads(%[[NUM_THREADS]] : i32)
  ! DEVICE-SAME: num_threads({{.*}})

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  ! BOTH-NEXT: omp.loop_nest

  ! HOST-SAME: (%{{.*}}) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: host_eval({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo() !< Prevents this from being SPMD.

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.parallel
  ! BOTH-SAME: num_threads({{.*}})
  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.wsloop
  ! BOTH-NEXT: omp.simd
  !$omp distribute parallel do simd num_threads(1)
  do i=1,10
    call foo()
  end do
  !$omp end distribute parallel do simd
  !$omp end teams
end subroutine distribute_parallel_do_simd

! BOTH-LABEL: func.func @_QPdistribute
subroutine distribute()
  ! BOTH: omp.target
  
  ! HOST-SAME: host_eval(%{{.*}} -> %[[LB:.*]], %{{.*}} -> %[[UB:.*]], %{{.*}} -> %[[STEP:.*]] : i32, i32, i32)
  
  ! DEVICE-NOT: host_eval({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.loop_nest

  ! HOST-SAME: (%{{.*}}) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  !$omp distribute
  do i=1,10
    call foo()
  end do
  !$omp end distribute
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: host_eval({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo() !< Prevents this from being Generic-SPMD.

  ! BOTH: omp.distribute
  !$omp distribute
  do i=1,10
    call foo()
  end do
  !$omp end distribute
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.distribute
  !$omp distribute
  do i=1,10
    call foo()
  end do
  !$omp end distribute
  !$omp end teams
end subroutine distribute

! BOTH-LABEL: func.func @_QPdistribute_simd
subroutine distribute_simd()
  ! BOTH: omp.target
  
  ! HOST-SAME: host_eval(%{{.*}} -> %[[LB:.*]], %{{.*}} -> %[[UB:.*]], %{{.*}} -> %[[STEP:.*]] : i32, i32, i32)
  
  ! DEVICE-NOT: host_eval({{.*}})
  ! DEVICE-SAME: {

  ! BOTH: omp.teams
  !$omp target teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.simd
  ! BOTH-NEXT: omp.loop_nest

  ! HOST-SAME: (%{{.*}}) : i32 = (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]])
  !$omp distribute simd
  do i=1,10
    call foo()
  end do
  !$omp end distribute simd
  !$omp end target teams

  ! BOTH: omp.target
  ! BOTH-NOT: host_eval({{.*}})
  ! BOTH-SAME: {
  ! BOTH: omp.teams
  !$omp target teams
  call foo() !< Prevents this from being Generic-SPMD.

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.simd
  !$omp distribute simd
  do i=1,10
    call foo()
  end do
  !$omp end distribute simd
  !$omp end target teams

  ! BOTH: omp.teams
  !$omp teams

  ! BOTH: omp.distribute
  ! BOTH-NEXT: omp.simd
  !$omp distribute simd
  do i=1,10
    call foo()
  end do
  !$omp end distribute simd
  !$omp end teams
end subroutine distribute_simd
