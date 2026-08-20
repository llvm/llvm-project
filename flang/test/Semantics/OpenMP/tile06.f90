! Testing the Semantics of tile
!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51


subroutine nonrectangular_loop_lb
  implicit none
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: None of the loops affected by TILE can be non-rectangular
  !$omp tile sizes(2,2)
  do i = 1, 5
    !BECAUSE: The upper bound of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 1, i
      print *, i, j
    end do
  end do
end subroutine


subroutine nonrectangular_loop_ub
  implicit none
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: None of the loops affected by TILE can be non-rectangular
  !$omp tile sizes(2,2)
  do i = 1, 5
    !BECAUSE: The upper bound of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 1, i
      print *, i, j
    end do
  end do
end subroutine


subroutine nonrectangular_loop_step
  implicit none
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: None of the loops affected by TILE can be non-rectangular
  !$omp tile sizes(2,2)
  do i = 1, 5
    !BECAUSE: The iteration step of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 1, 42, i
      print *, i, j
    end do
  end do
end subroutine
