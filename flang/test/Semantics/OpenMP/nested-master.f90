! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! Various checks with the nesting of MASTER construct

program omp_nest_master
  integer i, k, j
  k = 0;

  !$omp do
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    j = j -1
    !$omp end master
  end do

  !$omp sections 
  !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
  !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    do i = 1, 10
      k = k + 1
    end do
    !$omp end master
  !$omp end sections

  !$omp single 
  !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
  !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    do i = 1, 10
      k = k + 1
    end do
    !$omp end master
  !$omp end single



  !$omp task
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    j = j -1
    !$omp end master
  end do
  !$omp end task

  !$omp taskloop
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    j = j -1
    !$omp end master
  end do
  !$omp end taskloop

  !$omp target parallel do simd
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct, the `SCAN` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
    !$omp master
    j = j -1
    !$omp end master
  end do
  !$omp end target parallel do simd

  !$omp critical
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !$omp master
    j = j -1
    !$omp end master
  end do
  !$omp end critical

  !$omp ordered
  do i = 1, 10
    k = k + 1
    !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
    !$omp master
    j = j -1
    !$omp end master
  end do
  !$omp end ordered

  !$omp ordered
  do i = 1, 10
    !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
    !$omp teams
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
      !$omp master
      j = j -1
      !$omp end master
    end do
    !$omp end distribute
    !$omp end teams
  end do
  !$omp end ordered

  !$omp critical
  do i = 1, 10
    !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
    !$omp teams
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
      !$omp master
      j = j -1
      !$omp end master
    end do
    !$omp end distribute
    !$omp end teams
  end do
  !$omp end critical

  !$omp taskloop
  do i = 1, 10
    !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
    !$omp teams
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
      !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
      !$omp master
      j = j -1
      !$omp end master
    end do
    !$omp end distribute
    !$omp end teams
  end do
  !$omp end taskloop

  !$omp task
  do i = 1, 10
    !ERROR: TEAMS region can only be strictly nested within the implicit parallel region or TARGET region
    !$omp teams
    !$omp distribute
    do k =1, 10
      print *, "hello"
      !WARNING: OpenMP directive MASTER has been deprecated, please use MASKED instead.
      !ERROR: `MASTER` region may not be closely nested inside of `WORKSHARING`, `LOOP`, `TASK`, `TASKLOOP`, or `ATOMIC` region.
      !$omp master
      j = j -1
      !$omp end master
    end do
    !$omp end distribute
    !$omp end teams
  end do
  !$omp end task

end program omp_nest_master
