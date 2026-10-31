! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51
! OpenMP 5.1/5.2: only an ORDERED clause *with* an argument requires perfect
! nesting. A bare ORDERED clause does not, unlike in 5.0 (see
! ordered-nesting-omp50.f90).

! Bare ORDERED with COLLAPSE(2), imperfectly nested: valid in 5.1, since the
! bare ORDERED clause does not impose perfect nesting.
subroutine bare_ordered_collapse_imperfect
  integer i, j

  !$omp do collapse(2) ordered
  do i = 1, 10
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine

! ORDERED(2) with an argument still requires perfect nesting.
subroutine ordered_arg_imperfect
  integer i, j

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
