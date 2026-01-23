! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
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
      !CHECK: invalid branch leaving an OpenMP structured block
      goto 10
    end do
  end do
  !$omp end do

  !CHECK: Outside the enclosing DO directive
  10 stop

end program omp_do
