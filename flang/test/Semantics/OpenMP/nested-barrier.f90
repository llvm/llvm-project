! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! Various checks with the nesting of BARRIER construct

program omp_nest_barrier
  integer i, k, j
  k = 0;

  !$omp do
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do

  !$omp do simd
  do i = 1, 10
    k = k + 1
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct, the `SCAN` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do

  !$omp parallel do
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do

  !$omp parallel do simd
  do i = 1, 10
    k = k + 1
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct, the `SCAN` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do

  !$omp parallel
  do i = 1, 10
    k = k + 1
    !$omp barrier
    j = j -1
  end do
  !$omp end parallel

  !$omp task
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do
  !$omp end task

  !$omp taskloop
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do
  !$omp end taskloop

  !$omp critical
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do
  !$omp end critical

  !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
  !$omp master
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do
  !$omp end master

  !$omp ordered
  do i = 1, 10
    k = k + 1
    !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
    !$omp barrier
    j = j -1
  end do
  !$omp end ordered

  !$omp ordered
  do i = 1, 10
    !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
      !$omp barrier
      j = j -1
    end do
    !$omp end distribute
  end do
  !$omp end ordered

  !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
  !$omp master
  do i = 1, 10
    !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
      !$omp barrier
      j = j -1
    end do
    !$omp end distribute
  end do
  !$omp end master

  !$omp critical
  do i = 1, 10
    !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
      !$omp barrier
      j = j -1
    end do
    !$omp end distribute
  end do
  !$omp end critical

  !$omp taskloop
  do i = 1, 10
    !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
      !$omp barrier
      j = j -1
    end do
    !$omp end distribute
  end do
  !$omp end taskloop

  !$omp task
  do i = 1, 10
    !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !ERROR: `BARRIER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`,`CRITICAL`, `ORDERED`, `ATOMIC` or `MASTER` region.
      !$omp barrier
      j = j -1
    end do
    !$omp end distribute
  end do
  !$omp end task

end program omp_nest_barrier
