! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.15.4.1 copyin Clause
! A list item that appears in a copyin clause must be threadprivate

program omp_copyin

  integer :: i
  integer, save :: j, k
  integer :: a(10), b(10)

  !$omp threadprivate(j, k)

  j = 20
  k = 10

  !$omp parallel do copyin(j, k)
  do i = 1, 10
    a(i) = k + i
    b(i) = j + i
  end do
  !$omp end parallel do

  print *, a, b

end program omp_copyin
