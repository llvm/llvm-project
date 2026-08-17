! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=60

! Starting with OpenMP 6.0, directives with the "pure" property are allowed
! in a DO CONCURRENT construct; other directives are still rejected.

module m
contains
  subroutine do_concurrent_valid(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i
    do concurrent (i = 1:n)
      !$omp nothing
      a(i) = a(i) + 1
    end do
  end subroutine

  subroutine do_concurrent_bad(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i
    do concurrent (i = 1:n)
      !ERROR: The OpenMP directive 'BARRIER' is not allowed in a DO CONCURRENT construct
      !$omp barrier
      !ERROR: The OpenMP directive 'PARALLEL' is not allowed in a DO CONCURRENT construct
      !$omp parallel
      !$omp end parallel
      !ERROR: The OpenMP directive 'ATOMIC' is not allowed in a DO CONCURRENT construct
      !$omp atomic
      a(i) = a(i) + 1
    end do
  end subroutine

  ! A plain DO nested in DO CONCURRENT is still part of its body.
  subroutine do_concurrent_nested_do(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i, j
    do concurrent (i = 1:n)
      do j = 1, n
        !ERROR: The OpenMP directive 'TASK' is not allowed in a DO CONCURRENT construct
        !$omp task
        !$omp end task
      end do
      a(i) = a(i) + 1
    end do
  end subroutine
end module
