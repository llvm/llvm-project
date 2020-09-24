! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct

program omp_do
  integer i, j, k
  !$omp do collapse(2)
  do i = 1, 10
    !ERROR: CYCLE statement to non-innermost collapsed !$OMP DO loop
    if (i .lt. 5) cycle
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do

end program omp_do
