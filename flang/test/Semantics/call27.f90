! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Catch NULL() actual argument association with allocatable dummy argument
program test
  real, allocatable :: a
  !ERROR: NULL() actual argument 'NULL()' may not be associated with allocatable dummy argument dummy argument 'a=' that is INTENT(OUT) or INTENT(IN OUT)
  call foo0(null())
  !WARNING: NULL() actual argument 'NULL()' should not be associated with allocatable dummy argument dummy argument 'a=' without INTENT(IN)
  call foo1(null())
  !PORTABILITY: Allocatable dummy argument 'a=' is associated with NULL()
  call foo2(null())
  call foo3(null()) ! ok
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
  !BECAUSE: 'null(mold=a)' is a null pointer
  call foo0(null(mold=a))
  !WARNING: A null pointer should not be associated with allocatable dummy argument 'a=' without INTENT(IN)
  call foo1(null(mold=a))
  !PORTABILITY: Allocatable dummy argument 'a=' is associated with a null pointer
  call foo2(null(mold=a))
  call foo3(null(mold=a)) ! ok
 contains
  subroutine foo0(a)
    real, allocatable, intent(in out) :: a
  end subroutine
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
