! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror -pedantic
! Confirm a portability warning on use of a procedure binding apart from a call
module m
  type t
   contains
    procedure :: sub
  end type
 contains
  subroutine sub(x)
    class(t), intent(in) :: x
  end subroutine
end module

program test
  use m
  procedure(sub), pointer :: p
  type(t) x
  !PORTABILITY: Procedure binding 'sub' used as target of a pointer assignment
  p => x%sub
  !PORTABILITY: Procedure binding 'sub' passed as an actual argument
  call sub2(x%sub)
 contains
  subroutine sub2(s)
    procedure(sub) s
  end subroutine
end
