! RUN: %S/test_errors.sh %s %t %f18 -fopenmp
! XFAIL:*

! OpenMP Version 4.5
! 2.7.1 Loop Construct
! No statement in the associated loops other than the DO statements
! can cause a branch out of the loops

program omp_do
  integer i, j, k

  !$omp do
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
      !ERROR: invalid branch to/from OpenMP structured block
      goto 10
    end do
  end do
  !$omp end do

  10 stop

end program omp_do
