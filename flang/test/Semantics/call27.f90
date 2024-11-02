! RUN: %python %S/test_errors.py %s %flang_fc1
! Catch NULL() actual argument association with allocatable dummy argument
program test
  !ERROR: Null actual argument 'NULL()' may not be associated with allocatable dummy argument 'a='
  call foo1(null())
  !ERROR: Null actual argument 'NULL()' may not be associated with allocatable dummy argument 'a='
  call foo2(null()) ! perhaps permissible later on user request
  call foo3(null()) ! ok
 contains
  subroutine foo1(a)
    real, allocatable :: a
  end subroutine
  subroutine foo2(a)
    real, allocatable, intent(in) :: a
  end subroutine
  subroutine foo3(a)
    real, allocatable, optional :: a
  end subroutine
end
