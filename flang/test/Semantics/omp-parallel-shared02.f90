!RUN: %S/test_errors.sh %s %t %flang -fopenmp
!REQUIRES: shell
! OpenMP Version 4.5
! 2.15.3.2 parallel shared Clause
program omp_parallel_shared
  integer :: i, j, a(10), b(10), c(10)
  integer :: k = 10
  integer :: array(10)

  do i = 1, 10
    array(i) = i
  end do

  !ERROR: A variable that is part of another variable (as an array or structure element) cannot appear in a PRIVATE or SHARED clause or on the ALLOCATE directive.
  !$omp parallel shared(array(i))
  do i = 1, 10
    c(i) = a(i) + b(i) + k
    array(i) = k
  end do
  !$omp end parallel
end program omp_parallel_shared
