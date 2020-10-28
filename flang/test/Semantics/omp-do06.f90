! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL:*

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The ordered clause must be present on the loop construct if any ordered
! region ever binds to a loop region arising from the loop construct.

program omp_do
  integer i, j, k

  !$omp do
  do i = 1, 10
    !ERROR: ‘ordered’ region inside a loop region without an ordered clause.
    !$omp ordered
    call my_func()
    !$omp end ordered
  end do
  !$omp end do

end program omp_do
