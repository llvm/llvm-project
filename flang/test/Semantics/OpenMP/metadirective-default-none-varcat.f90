!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine default_none_varcat(n, a)
integer :: n, i
integer, dimension(:), pointer :: p
integer, target :: a(n)
integer :: x
p => a
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none: pointer) shared(n))
  do i = 1, n
    !ERROR: The DEFAULT(NONE) clause requires that 'p' must be listed in a data-sharing attribute clause
    p(i) = i
    x = i
  end do
end subroutine

subroutine call_in_loop(n, a)
  integer :: n, a(n), i
  external :: foo
  !$omp metadirective &
  !$omp& when(implementation={vendor(llvm)}: &
  !$omp& parallel do default(none: scalar) shared(n, a)) otherwise(nothing)
  do i = 1, n
    call foo(a(i))
  end do
end subroutine
