! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

subroutine dont_crash(values)
  implicit none
  integer, parameter :: n = 100
  real :: values(n)
  integer :: i
  !ERROR: No explicit type declared for 'sum'
  sum = 0
  !ERROR: No explicit type declared for 'sum'
  !$omp parallel do reduction(+:sum)
  do i = 1, n
  !ERROR: No explicit type declared for 'sum'
  !ERROR: No explicit type declared for 'sum'
     sum = sum + values(i)
  end do
end subroutine dont_crash

