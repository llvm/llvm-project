! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=51

! In OpenMP 5.1, only SIMD and declarative directives are permitted in a PURE
! procedure. The metadirective, assumption, nothing, error, and
! loop-transforming directives were not added to that list until OpenMP 5.2, so
! they are diagnosed here with a version-specific message. Directives that never
! have the "pure" property (e.g. parallel, task) are rejected outright.

module m
contains
  pure subroutine pure_ok(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i
    ! SIMD and declarative directives are allowed in every version.
    !$omp declare reduction(myadd : integer : omp_out = omp_out + omp_in) &
    !$omp& initializer(omp_priv = 0)
    !$omp simd
    do i = 1, n
      a(i) = a(i) + 1
    end do
  end subroutine

  pure subroutine pure_bad(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i
    !ERROR: The OpenMP directive 'NOTHING' is not allowed in a PURE procedure in OpenMP v5.1, try -fopenmp-version=52
    !$omp nothing
    !ERROR: The OpenMP directive 'METADIRECTIVE' is not allowed in a PURE procedure in OpenMP v5.1, try -fopenmp-version=52
    !$omp metadirective when(user={condition(.true.)}: nothing)
    !ERROR: The OpenMP directive 'TILE' is not allowed in a PURE procedure in OpenMP v5.1, try -fopenmp-version=52
    !$omp tile sizes(4)
    do i = 1, n
      a(i) = a(i) + 1
    end do
    !ERROR: The OpenMP directive 'UNROLL' is not allowed in a PURE procedure in OpenMP v5.1, try -fopenmp-version=52
    !$omp unroll partial(2)
    do i = 1, n
      a(i) = a(i) * 2
    end do
    !ERROR: The OpenMP directive 'PARALLEL' is not allowed in a PURE procedure
    !$omp parallel
    !$omp end parallel
    !ERROR: The OpenMP directive 'TASK' is not allowed in a PURE procedure
    !$omp task
    !$omp end task
  end subroutine
end module
