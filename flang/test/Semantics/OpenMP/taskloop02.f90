! RUN: not %flang -fsyntax-only -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! 2.9.2 taskloop Construct
! Invalid entry to OpenMP structured block.

program omp_taskloop
  integer i , j

  !CHECK: invalid branch into an OpenMP structured block
  goto 10

  !$omp taskloop private(j) grainsize(500) nogroup
  do i=1,10000
    do j=1,i
      !CHECK: In the enclosing TASKLOOP directive branched into
      10 call loop_body(i, j)
    end do
  end do
  !$omp end taskloop

end program omp_taskloop
