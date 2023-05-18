! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s --allow-empty

! Regression test: don't emit a bogus error about an invalid specification expression
! in the declaration of a binding

module m
  type :: t
    integer :: n
   contains
    !CHECK-NOT: Invalid specification expression
    procedure :: binding => func
  end type
 contains
  function func(x)
    class(t), intent(in) :: x
    character(len=x%n) :: func
    func = ' '
  end function
end module
