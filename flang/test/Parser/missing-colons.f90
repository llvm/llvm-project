! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
module m
  type t
   contains
!CHECK: portability: type-bound procedure statement should have '::' if it has '=>'
    procedure p => sub
  end type
 contains
  subroutine sub(x)
    class(t), intent(in) :: x
  end subroutine
end module

