! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! OpenMP 5.2: doacross loop nests (those with ordered doacross(sink/source)
! constructs) require perfect nesting.

! ordered(2) without doacross directives: imperfect nesting is valid in 5.2.
subroutine ordered_no_doacross_imperfect
  integer i, j

  !$omp do ordered(2)
  do i = 1, 10
    print *, i
    do j = 1, 10
      print *, i, j
    end do
  end do
  !$omp end do
end subroutine

! Perfectly nested doacross: valid.
subroutine doacross_perfect
  integer i, j

  !$omp do ordered(2)
  do i = 1, 10
    do j = 1, 10
      !$omp ordered doacross(sink: i-1, j)
      print *, i, j
      !$omp ordered doacross(source)
    end do
  end do
  !$omp end do
end subroutine

! Imperfectly nested doacross: invalid in 5.2.
subroutine doacross_imperfect
  integer i, j

  !ERROR: This construct requires a perfect nest of depth 2, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 2
  !$omp do ordered(2)
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      !$omp ordered doacross(sink: i-1, j)
      print *, i, j
      !$omp ordered doacross(source)
    end do
  end do
  !$omp end do
end subroutine

! collapse(2) ordered(3) without doacross: imperfect nesting is valid.
subroutine collapse_ordered_no_doacross_imperfect
  integer i, j, k

  !$omp do collapse(2) ordered(3)
  do i = 1, 10
    print *, i
    do j = 1, 10
      do k = 1, 10
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do
end subroutine

! Doacross with collapse: ordered(N) controls depth when N > collapse.
subroutine doacross_collapse
  integer i, j, k

  !ERROR: This construct requires a perfect nest of depth 3, but the associated nest is a perfect nest of depth 1
  !BECAUSE: ORDERED clause was specified with argument 3
  !$omp do collapse(2) ordered(3)
  do i = 1, 10
    !BECAUSE: This code prevents perfect nesting
    print *, i
    do j = 1, 10
      do k = 1, 10
        !$omp ordered doacross(sink: i-1, j, k)
        print *, i, j, k
        !$omp ordered doacross(source)
      end do
    end do
  end do
  !$omp end do
end subroutine

! Doacross inside a nested OpenMP region should not force perfect nesting on
! the outer loop. The doacross binds to the inner loop, not the outer one.
subroutine doacross_in_nested_region
  integer i, j, k

  !$omp do collapse(2)
  do i = 1, 10
    print *, i
    do j = 1, 10
      !$omp parallel
      !$omp do ordered(1)
      do k = 1, 10
        !$omp ordered doacross(source)
        print *, k
      end do
      !$omp end do
      !$omp end parallel
    end do
  end do
  !$omp end do
end subroutine
