! RUN: %flang_fc1 -fdebug-unparse-no-sema -fopenmp %s | FileCheck %s
! Check Unparsing of OpenMP IF clause

program if_unparse
  logical :: cond
  integer :: i

  ! CHECK: !$OMP TARGET UPDATE
  ! CHECK-SAME: IF(cond)
  !$omp target update if(cond)

  ! CHECK: !$OMP TARGET UPDATE
  ! CHECK-SAME: IF(TARGETUPDATE:cond)
  !$omp target update if(target update: cond)
  
  ! CHECK: !$OMP TARGET UPDATE
  ! CHECK-SAME: IF(TARGETUPDATE:cond)
  !$omp target update if(targetupdate: cond)

  ! CHECK: !$OMP TARGET ENTER DATA
  ! CHECK-SAME: IF(TARGETENTERDATA:cond)
  !$omp target enter data map(to: i) if(target enter data: cond)

  ! CHECK: !$OMP TARGET EXIT DATA
  ! CHECK-SAME: IF(TARGETEXITDATA:cond)
  !$omp target exit data map(from: i) if(target exit data: cond)

  ! CHECK: !$OMP TARGET DATA
  ! CHECK-SAME: IF(TARGETDATA:cond)
  !$omp target data map(tofrom: i) if(target data: cond)
  !$omp end target data

  ! CHECK: !$OMP TARGET
  ! CHECK-SAME: IF(TARGET:cond)
  !$omp target if(target: cond)
  !$omp end target

  ! CHECK: !$OMP TEAMS
  ! CHECK-SAME: IF(TEAMS:cond)
  !$omp teams if(teams: cond)
  !$omp end teams

  ! CHECK: !$OMP PARALLEL DO SIMD
  ! CHECK-SAME: IF(PARALLEL:i<10) IF(SIMD:.FALSE.)
  !$omp parallel do simd if(parallel: i < 10) if(simd: .false.)
  do i = 1, 10
  end do
  !$omp end parallel do simd

  ! CHECK: !$OMP TASK
  ! CHECK-SAME: IF(TASK:cond)
  !$omp task if(task: cond)
  !$omp end task

  ! CHECK: !$OMP TASKLOOP
  ! CHECK-SAME: IF(TASKLOOP:cond)
  !$omp taskloop if(taskloop: cond)
  do i = 1, 10
  end do
  !$omp end taskloop
end program if_unparse
