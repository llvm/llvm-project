! Test invalid SIMPLE procedure attribute combinations and procedure argument
! compatibility checks.
! RUN: %python %S/test_errors.py %s %flang_fc1

! --- CASE 1: SIMPLE + IMPURE conflict
!ERROR: A procedure may not have both the SIMPLE and IMPURE attributes
simple impure subroutine bug()
end subroutine

! --- CASE 2: IMPURE + SIMPLE conflict
!ERROR: A procedure may not have both the SIMPLE and IMPURE attributes
impure simple subroutine bug2()
end subroutine

! --- CASE 3: PURE does NOT satisfy SIMPLE
module pure_not_enough_for_simple
  abstract interface
    simple subroutine simple_proc()
    end subroutine
  end interface
contains
  subroutine needs_simple(p)
    procedure(simple_proc) :: p
  end subroutine
  pure subroutine pure_only()
  end subroutine
  subroutine test()
    !ERROR: Actual procedure argument has interface incompatible with dummy argument 'p=': incompatible procedure attributes: Simple
    call needs_simple(pure_only)
  end subroutine
end module

! --- CASE 4: Neither PURE nor SIMPLE does NOT satisfy PURE
module m
  implicit none
  abstract interface
    pure integer function iproc(x)
      integer, intent(in) :: x
    end function
  end interface
contains
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
