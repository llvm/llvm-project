! This is the negative/control case for simple-satisfies-pure.f90.
! It verifies that a procedure which is neither PURE nor SIMPLE is rejected
! when passed to a dummy argument that requires a PURE procedure.
! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  implicit none

  abstract interface
    pure integer function iproc(x)
      integer, intent(in) :: x
    end function
  end interface

contains

  ! Neither PURE nor SIMPLE -> must NOT satisfy PURE requirements
  integer function impure(x)
    integer, intent(in) :: x
    impure = x
  end function

  pure integer function apply_pure(f, x)
    procedure(iproc) :: f
    integer, intent(in) :: x
    apply_pure = f(x)
  end function

  integer function test()
    !ERROR: Actual procedure argument has interface incompatible with dummy argument 'f=': incompatible procedure attributes: Pure
    test = apply_pure(impure, 1)
  end function
end module
