! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=52

! Prior to OpenMP 6.0, no OpenMP directive is allowed inside a DO CONCURRENT
! construct, regardless of whether it would otherwise have the "pure"
! property (e.g. SIMD, which has been "pure" since OpenMP 4.5).

module m
contains
  subroutine do_concurrent_bad(a, n)
    integer, intent(in) :: n
    integer, intent(inout) :: a(n)
    integer :: i, j
    do concurrent (i = 1:n)
      !ERROR: The OpenMP directive 'SIMD' is not allowed in a DO CONCURRENT construct
      !$omp simd
      do j = 1, n
      end do
      !ERROR: The OpenMP directive 'BARRIER' is not allowed in a DO CONCURRENT construct
      !$omp barrier
    end do
  end subroutine
end module
