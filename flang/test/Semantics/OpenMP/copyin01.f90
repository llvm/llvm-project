! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.15.4.1 copyin Clause
! A list item that appears in a copyin clause must be threadprivate

program omp_copyin

  integer :: i
  integer, save :: k
  integer :: a(10), b(10)
  common /cmn/ j

  k = 10

  !ERROR: Non-THREADPRIVATE object 'k' in COPYIN clause
  !$omp parallel do copyin(k)
  do i = 1, 10
    a(i) = k + i
    j = j + a(i)
  end do
  !$omp end parallel do

  print *, a

  !ERROR: Non-THREADPRIVATE object 'j' in COPYIN clause
  !$omp parallel do copyin(/cmn/)
  do i = 1, 10
    b(i) = a(i) + j
  end do
  !$omp end parallel do

  print *, b

end program omp_copyin
