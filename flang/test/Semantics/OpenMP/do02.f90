! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! Exit statement terminating !$OMP DO loop

program omp_do
  integer i, j, k

  !$omp do
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
    !ERROR: EXIT statement terminating !$OMP DO loop
    exit
  end do
  !$omp end do

end program omp_do
