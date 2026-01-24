! This test verifies that SIMPLE procedures satisfy PURE requirements.
! (i.e. anywhere the language requires a PURE procedure, a SIMPLE one is accepted)
! RUN: %flang_fc1 -fsyntax-only %s

module m
  implicit none

  abstract interface
    ! Dummy procedure explicitly requires PURE
    pure integer function pure_iface(x)
      integer, intent(in) :: x
    end function
  end interface

contains

  ! SIMPLE procedure (should be accepted where PURE is required)
  simple integer function simple_impl(x)
    integer, intent(in) :: x
    simple_impl = x
  end function

  pure integer function apply_requires_pure(f, x)
    procedure(pure_iface) :: f
    integer, intent(in) :: x
    apply_requires_pure = f(x)
  end function

  integer function test()
    test = apply_requires_pure(simple_impl, 1)
  end function
end module

