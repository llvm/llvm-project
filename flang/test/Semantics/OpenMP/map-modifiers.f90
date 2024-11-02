!RUN: %python %S/../test_errors.py %s %flang -fopenmp -Werror

subroutine f10(x)
  integer :: x
!PORTABILITY: the specification of modifiers without comma separators for the 'MAP' clause has been deprecated
  !$omp target map(always, present close, to: x)
  x = x + 1
  !$omp end target
end

subroutine f11(x)
  integer :: x
!PORTABILITY: the specification of modifiers without comma separators for the 'MAP' clause has been deprecated
  !$omp target map(always, present, close to: x)
  x = x + 1
  !$omp end target
end

