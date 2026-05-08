!RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -fopenmp-version=50

module m

interface
  subroutine f(x, y)
    integer, allocatable :: x
    integer :: y
    !$omp declare simd(f) linear(ref(x) : 1) linear(uval(y))
  end
end interface

contains

subroutine g
  integer :: i
  !ERROR: Clause LINEAR is not allowed if clause ORDERED appears on the DO directive
  !ERROR: Loop iteration variable with a predetermined data sharing attribute cannot appear in a LINEAR clause
  !$omp do ordered(1) linear(i)
  !BECAUSE: 'i' is an iteration variable of an affected loop
  do i = 1, 10
  end do
end

end module
