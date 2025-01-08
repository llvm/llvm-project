! RUN: %python %S/../test_errors.py %s %flang -fopenmp

program main
  implicit none
  integer :: N
  integer :: i
  real(8) :: a(256), b(256)
  N = 256

  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute simd
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end distribute simd

  !$omp target parallel device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !ERROR: At most one DEVICE clause can appear on the TARGET PARALLEL directive
  !$omp target parallel device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !$omp target parallel defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !ERROR: 'variable-category' modifier is required
  !$omp target parallel defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET PARALLEL directive
  !$omp target parallel defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !$omp target parallel map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !ERROR: COPYIN clause is not allowed on the TARGET PARALLEL directive
  !ERROR: Non-THREADPRIVATE object 'a' in COPYIN clause
  !$omp target parallel copyin(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel

  !$omp target parallel do device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !ERROR: At most one DEVICE clause can appear on the TARGET PARALLEL DO directive
  !$omp target parallel do device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !$omp target parallel do defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !ERROR: 'variable-category' modifier is required
  !$omp target parallel do defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET PARALLEL DO directive
  !$omp target parallel do defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !$omp target parallel do map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !ERROR: COPYIN clause is not allowed on the TARGET PARALLEL DO directive
  !ERROR: Non-THREADPRIVATE object 'a' in COPYIN clause
  !$omp target parallel do copyin(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target parallel do

  !$omp target teams map(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !$omp target teams device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: At most one DEVICE clause can appear on the TARGET TEAMS directive
  !$omp target teams device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: SCHEDULE clause is not allowed on the TARGET TEAMS directive
  !$omp target teams schedule(static)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !$omp target teams defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: 'variable-category' modifier is required
  !$omp target teams defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET TEAMS directive
  !$omp target teams defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !$omp target teams num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: At most one NUM_TEAMS clause can appear on the TARGET TEAMS directive
  !$omp target teams num_teams(2) num_teams(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp target teams num_teams(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: At most one THREAD_LIMIT clause can appear on the TARGET TEAMS directive
  !$omp target teams thread_limit(2) thread_limit(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: The parameter of the THREAD_LIMIT clause must be a positive integer expression
  !$omp target teams thread_limit(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: At most one DEFAULT clause can appear on the TARGET TEAMS directive
  !$omp target teams default(shared) default(private)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !$omp target teams num_teams(2) defaultmap(tofrom:scalar)
  do i = 1, N
      a(i) = 3.14
  enddo
  !$omp end target teams

  !$omp target teams map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams

  !ERROR: Only the TO, FROM, TOFROM, ALLOC map types are permitted for MAP clauses on the TARGET TEAMS directive
  !$omp target teams map(delete:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams


  !$omp target teams distribute map(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: At most one DEVICE clause can appear on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: 'variable-category' modifier is required
  !$omp target teams distribute defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: At most one NUM_TEAMS clause can appear on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute num_teams(2) num_teams(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp target teams distribute num_teams(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: At most one THREAD_LIMIT clause can appear on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute thread_limit(2) thread_limit(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: The parameter of the THREAD_LIMIT clause must be a positive integer expression
  !$omp target teams distribute thread_limit(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: At most one DEFAULT clause can appear on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute default(shared) default(private)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute num_teams(2) defaultmap(tofrom:scalar)
  do i = 1, N
      a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !ERROR: Only the TO, FROM, TOFROM, ALLOC map types are permitted for MAP clauses on the TARGET TEAMS DISTRIBUTE directive
  !$omp target teams distribute map(delete:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute

  !$omp target teams distribute parallel do device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: At most one DEVICE clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: 'variable-category' modifier is required
  !$omp target teams distribute parallel do defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: At most one NUM_TEAMS clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do num_teams(2) num_teams(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp target teams distribute parallel do num_teams(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: At most one THREAD_LIMIT clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do thread_limit(2) thread_limit(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: The parameter of the THREAD_LIMIT clause must be a positive integer expression
  !$omp target teams distribute parallel do thread_limit(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: At most one DEFAULT clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do default(shared) default(private)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do num_teams(2) defaultmap(tofrom:scalar)
  do i = 1, N
      a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do

  !ERROR: Only the TO, FROM, TOFROM, ALLOC map types are permitted for MAP clauses on the TARGET TEAMS DISTRIBUTE PARALLEL DO directive
  !$omp target teams distribute parallel do map(delete:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do


  !$omp target teams distribute parallel do simd map(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd device(0)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: At most one DEVICE clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd device(0) device(1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: 'variable-category' modifier is required
  !$omp target teams distribute parallel do simd defaultmap(tofrom)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: At most one DEFAULTMAP clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd defaultmap(tofrom:scalar) defaultmap(tofrom:scalar)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd num_teams(3) thread_limit(10) default(shared) private(i) shared(a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: At most one NUM_TEAMS clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd num_teams(2) num_teams(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: The parameter of the NUM_TEAMS clause must be a positive integer expression
  !$omp target teams distribute parallel do simd num_teams(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: At most one THREAD_LIMIT clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd thread_limit(2) thread_limit(3)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: The parameter of the THREAD_LIMIT clause must be a positive integer expression
  !$omp target teams distribute parallel do simd thread_limit(-1)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: At most one DEFAULT clause can appear on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd default(shared) default(private)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd num_teams(2) defaultmap(tofrom:scalar)
  do i = 1, N
      a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute parallel do simd map(tofrom:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd

  !ERROR: Only the TO, FROM, TOFROM, ALLOC map types are permitted for MAP clauses on the TARGET TEAMS DISTRIBUTE PARALLEL DO SIMD directive
  !$omp target teams distribute parallel do simd map(delete:a)
  do i = 1, N
     a(i) = 3.14
  enddo
  !$omp end target teams distribute parallel do simd


end program main

