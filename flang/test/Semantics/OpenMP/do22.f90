! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check for existence of loop following a DO directive

subroutine do_imperfectly_nested_before
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
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

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: COLLAPSE clause was specified with argument 2
  !$omp do collapse(2)
  do i = 1, 10
    do j = 1, 10
      print *, i, j
    end do
    print *, i
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_lb
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp do collapse(2)
  do i = 1, 10
    do j = i, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_ub
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp do collapse(2)
  do i = 1, 10
    do j = 0, i
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine


subroutine do_nonrectangular_step
  integer i, j

  !ERROR: Trip count must be computable and invariant
  !$omp do collapse(2)
  do i = 1, 10
    do j = 1, 10, i
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine
