! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! Various checks with the ordered construct

SUBROUTINE LINEAR_GOOD(N)
  INTEGER N, i, j, a, b(10)
  !$omp target
  !$omp teams
  !$omp distribute parallel do simd linear(i) 
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target
END SUBROUTINE LINEAR_GOOD

SUBROUTINE LINEAR_BAD(N)
  INTEGER N, i, j, a, b(10)

  !$omp target
  !$omp teams
  !ERROR: Variable 'j' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !$omp distribute parallel do simd linear(j) 
  do i = 1, N
      a = 3.14
  enddo
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target

  !$omp target
  !$omp teams
  !ERROR: Variable 'j' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !ERROR: Variable 'b' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !$omp distribute parallel do simd linear(j) linear(b)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target 

  !$omp target
  !$omp teams
  !ERROR: Variable 'j' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !ERROR: Variable 'b' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !$omp distribute parallel do simd linear(j, b)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute parallel do simd
  !$omp end teams
  !$omp end target 

  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !ERROR: Variable 'j' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
  !$omp distribute simd linear(i,j)
   do i = 1, N
      do j = 1, N
         a = 3.14
      enddo
   enddo
   !$omp end distribute simd

   !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
   !ERROR: Variable 'j' not allowed in LINEAR clause, only loop iterator can be specified in LINEAR clause of a construct combined with DISTRIBUTE
   !$omp distribute simd linear(i,j) collapse(1)
   do i = 1, N
      do j = 1, N
         a = 3.14
      enddo
   enddo
   !$omp end distribute simd

   !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
   !$omp distribute simd linear(i,j) collapse(2)
   do i = 1, N
      do j = 1, N
         a = 3.14
      enddo
   enddo
   !$omp end distribute simd

END SUBROUTINE LINEAR_BAD
