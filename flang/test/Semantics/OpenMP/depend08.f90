! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine depend_iterator_noninteger_iv(x)
  integer :: x(10)
  !ERROR: The iterator variable must be of integer type
  !$omp task depend(iterator(real :: r = 1:3), in: x(int(r)))
  !$omp end task
end subroutine

! The range is nonempty; truncating the bounds to 64 bits would reverse them.
subroutine depend_iterator_wide_constants(x)
  integer :: x(3)
  !$omp task depend(iterator(integer(16) :: i = &
  !$omp&    9223372036854775807_16:9223372036854775809_16), &
  !$omp&    in: x(i - 9223372036854775806_16))
  !$omp end task
end subroutine
