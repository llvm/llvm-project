! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! 2.15.4.1 copyin Clause
! A list item that appears in a copyin clause must be threadprivate.
! Named variables appearing in a threadprivate common block may be specified
! It is not necessary to specify the whole common block.

program omp_copyin

  integer :: a(10), b(10)
  common /cmn/ j, k

  !$omp threadprivate(/cmn/)

  j = 20
  k = 10

  !$omp parallel copyin(/cmn/)
  a(:5) = k
  b(:5) = j
  !$omp end parallel

  j = j + k
  k = k * j

  !$omp parallel copyin(j, k)
  a(6:) = j
  b(6:) = k
  !$omp end parallel

  print *, a, b

end program omp_copyin
