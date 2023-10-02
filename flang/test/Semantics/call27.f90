! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Catch NULL() actual argument association with allocatable dummy argument
program test
  !ERROR: NULL() actual argument 'NULL()' may not be associated with allocatable dummy argument 'a=' without INTENT(IN)
  call foo1(null())
  !PORTABILITY: Allocatable dummy argument 'a=' is associated with NULL()
  call foo2(null())
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
