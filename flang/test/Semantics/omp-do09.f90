! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL: *

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The do-loop cannot be a DO WHILE or a DO loop without loop control.

program omp_do
  integer i, j, k
  i = 0

  !$omp do
  !ERROR: !$OMP DO cannot be a DO WHILE or DO without loop control
  do while (i .lt. 10)
    do j = 1, 10
      print *, "Hello"
    end do
    i = i + 1
  end do
  !$omp end do

end program omp_do
