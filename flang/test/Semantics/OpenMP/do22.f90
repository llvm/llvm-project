! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check for existence of loop following a DO directive

subroutine do_imperfectly_nested_before
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp do collapse(2)
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfectly_nested_behind
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp do collapse(2)
  do i = 1, 10
    do j = 1, 10
      print *, i, j
    end do
    !BECAUSE: This code prevents perfect nesting
    print *, i
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_lb
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: When SCHEDULE clause is present, none of the loops affected by DO can be non-rectangular
  !$omp do collapse(2) schedule(auto)
  do i = 1, 10
    !BECAUSE: The lower bound of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = i, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_ub
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: When SCHEDULE clause is present, none of the loops affected by DO can be non-rectangular
  !$omp do collapse(2) schedule(auto)
  do i = 1, 10
    !BECAUSE: The upper bound of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 0, i
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_step
  integer i, j

  !ERROR: This construct requires a rectangular loop nest, but the associated nest is not
  !BECAUSE: When SCHEDULE clause is present, none of the loops affected by DO can be non-rectangular
  !$omp do collapse(2) schedule(auto)
  do i = 1, 10
    !BECAUSE: The iteration step of the affected loop uses iteration variables of enclosing loops: 'i'
    do j = 1, 10, i
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine
