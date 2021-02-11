! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a threadprivate directive.

program omp_do
  integer i, j, k

  !$omp do firstprivate(i)
  !ERROR: !$OMP DO iteration variable i is not allowed in threadprivate
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end program omp_do
