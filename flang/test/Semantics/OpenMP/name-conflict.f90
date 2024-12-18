!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
module m

contains

subroutine foo1()
  integer :: baz1
!ERROR: 'baz1' must be a variable
!$omp parallel do shared(baz1)
  baz1: do i = 1, 100
  enddo baz1
!$omp end parallel do
end subroutine

subroutine foo2()
  !implicit baz2
!ERROR: 'baz2' must be a variable
!$omp parallel do shared(baz2)
  baz2: do i = 1, 100
  enddo baz2
!$omp end parallel do
end subroutine

end module m
