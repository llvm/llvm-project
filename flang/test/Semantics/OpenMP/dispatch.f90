! RUN: %python %S/../test_errors.py %s %flang -fopenmp

subroutine sb1
  integer :: r
  r = 1
  !ERROR: The DISPATCH construct does not contain a SUBROUTINE or FUNCTION
  !$omp dispatch nowait
  print *,r
end subroutine
subroutine sb2
  integer :: r
!ERROR: The DISPATCH construct is empty or contains more than one statement
  !$omp dispatch
  call foo()
  r = bar()
  !$omp end dispatch
contains
  subroutine foo
  end subroutine foo
  function bar
    integer :: bar
    bar = 2
  end function
end subroutine
