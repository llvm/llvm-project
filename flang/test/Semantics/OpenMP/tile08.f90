! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine do_concurrent
  implicit none
  integer i, j


  !$omp tile sizes(2,2)
  !ERROR: DO CONCURRENT loops cannot form part of a loop nest.
  do concurrent (i = 1:42, j = 1:42)
    print *, i, j
  end do
end subroutine
