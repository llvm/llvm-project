! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check OpenMP clause validity for the following directives:
!     2.10 Device constructs
program main

  real(8) :: arrayA(256), arrayB(256)
  integer :: N

  arrayA = 1.414
  arrayB = 3.14
  N = 256

  !$omp target map(arrayA)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target device(0)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: At most one DEVICE clause can appear on the TARGET directive
  !$omp target device(0) device(1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: SCHEDULE clause is not allowed on the TARGET directive
  !$omp target schedule(static)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target defaultmap(tofrom:scalar)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: The argument TOFROM:SCALAR must be specified on the DEFAULTMAP clause
  !$omp target defaultmap(tofrom)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET directive
  !$omp target defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target thread_limit(4)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: At most one THREAD_LIMIT clause can appear on the TARGET directive
  !$omp target thread_limit(4) thread_limit(8)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp teams num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !ERROR: At most one NUM_TEAMS clause can appear on the TEAMS directive
  !$omp teams num_teams(2) num_teams(3)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp teams num_teams(-1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !ERROR: At most one THREAD_LIMIT clause can appear on the TEAMS directive
  !$omp teams thread_limit(2) thread_limit(3)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !ERROR: The parameter of the THREAD_LIMIT clause must be a positive integer expression
  !$omp teams thread_limit(-1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !ERROR: At most one DEFAULT clause can appear on the TEAMS directive
  !$omp teams default(shared) default(private)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end teams

  !$omp target teams num_teams(2) defaultmap(tofrom:scalar)
  do i = 1, N
      a = 3.14
  enddo
  !$omp end target teams

  !$omp target map(tofrom:a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !ERROR: Only the TO, FROM, TOFROM, ALLOC map types are permitted for MAP clauses on the TARGET directive
  !$omp target map(delete:a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target

  !$omp target data device(0) map(to:a)
  do i = 1, N
    a = 3.14
  enddo
  !$omp end target data

  !ERROR: At least one of MAP clause must appear on the TARGET DATA directive
  !$omp target data device(0)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end target data

  !ERROR: The device expression of the DEVICE clause must be a positive integer expression
  !$omp target enter data map(alloc:A) device(-2)

  !ERROR: The device expression of the DEVICE clause must be a positive integer expression
  !$omp target exit data map(delete:A) device(-2)

  !ERROR: At most one IF clause can appear on the TARGET ENTER DATA directive
  !$omp target enter data map(to:a) if(.true.) if(.false.)

  !ERROR: Only the TO, ALLOC map types are permitted for MAP clauses on the TARGET ENTER DATA directive
  !$omp target enter data map(from:a)

  !$omp target exit data map(delete:a)

  !ERROR: At most one DEVICE clause can appear on the TARGET EXIT DATA directive
  !$omp target exit data map(from:a) device(0) device(1)

  !ERROR: Only the FROM, RELEASE, DELETE map types are permitted for MAP clauses on the TARGET EXIT DATA directive
  !$omp target exit data map(to:a)

  !$omp target
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end target

  !$omp target
  !$omp teams
  !$omp distribute
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end teams
  !$omp end target

  !$omp target
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !ERROR: At most one COLLAPSE clause can appear on the DISTRIBUTE directive
  !$omp distribute collapse(2) collapse(3)
  do i = 1, N
     do j = 1, N
        do k = 1, N
           a = 3.14
        enddo
     enddo
  enddo
  !$omp end distribute
  !$omp end target

  !$omp target
  !$omp teams
  !ERROR: At most one COLLAPSE clause can appear on the DISTRIBUTE directive
  !$omp distribute collapse(2) collapse(3)
  do i = 1, N
     do j = 1, N
        do k = 1, N
           a = 3.14
        enddo
     enddo
  enddo
  !$omp end distribute
  !$omp end teams
  !$omp end target

  !$omp target
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute dist_schedule(static, 2)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end target

  !$omp target
  !$omp teams
  !$omp distribute dist_schedule(static, 2)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end teams
  !$omp end target

  !$omp target
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !ERROR: At most one DIST_SCHEDULE clause can appear on the DISTRIBUTE directive
  !$omp distribute dist_schedule(static, 2) dist_schedule(static, 3)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end target

  !$omp target
  !$omp teams
  !ERROR: At most one DIST_SCHEDULE clause can appear on the DISTRIBUTE directive
  !$omp distribute dist_schedule(static, 2) dist_schedule(static, 3)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end teams
  !$omp end target

end program main
