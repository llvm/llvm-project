! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check for existence of loop following a DO directive

subroutine do_imperfectly_nested_before
  integer i, j

  ! Valid: print is allowed as CLN intervening code with collapse
  !$omp do collapse(2)
  do i = 1, 10
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfectly_nested_behind
  integer i, j

  ! Valid: print is allowed as CLN intervening code with collapse
  !$omp do collapse(2)
  do i = 1, 10
    do j = 1, 10
      print *, i, j
    end do
    print *, i
  end do
  !$omp end do
end subroutine


subroutine do_imperfectly_nested_scalar_assign
  integer i, j, x

  ! Valid: scalar assignment is allowed as CLN intervening code with collapse
  !$omp do collapse(2)
  do i = 1, 10
    x = i + 1
    do j = 1, 10
      print *, i, j, x
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfectly_nested_call
  integer i, j

  ! Valid: subroutine call is allowed as CLN intervening code with collapse
  !$omp do collapse(2)
  do i = 1, 10
    call sub(i)
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfectly_nested_multiple
  integer i, j, x

  ! Valid: multiple scalar statements are allowed as CLN intervening code
  !$omp do collapse(2)
  do i = 1, 10
    x = i * 2
    print *, x
    call sub(x)
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfect_collapse_bare_ordered
  integer i, j, x

  ! Valid: bare ORDERED does not require a perfect nest.
  !$omp do collapse(2) ordered
  do i = 1, 10
    x = 0
    do j = 1, 10
      !$omp ordered
      print *, i, j, x
      !$omp end ordered
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfect_ordered_requires_perfect
  integer i, j

  ! ordered(2) still requires perfect nesting at default OpenMP version
  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 2
  !$omp do ordered(2)
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfect_collapse_ordered_requires_perfect
  integer i, j, k

  ! collapse(2) ordered(3) requires perfect nesting at default OpenMP version because ordered(3) > collapse(2)
  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 3
  !$omp do collapse(2) ordered(3)
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      do k = 1, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do
end subroutine


subroutine do_imperfect_array_assign_invalid
  integer i, j
  integer :: a(10)

  ! Array assignment is invalid CLN intervening code
  !ERROR: This construct requires a nest of depth 2, but the associated nest is a nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp do collapse(2)
  do i = 1, 10
    !BECAUSE: The nest contains code that prevents it from being canonical at this nesting level
    a(:) = 0
    do j = 1, 10
      print *, i, j
    end do
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
