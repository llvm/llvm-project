! RUN: %python %S/test_errors.py %s %flang_fc1
module m
  type t
    integer n
   contains
    procedure :: assign1 => myassign, assign2 => myassign
    generic :: ASSIGNMENT(=) => assign1
    generic :: ASSIGNMENT(=) => assign2
  end type
 contains
  subroutine myassign(to, from)
    class(t), intent(out) :: to
    integer, intent(in) :: from
    to%n = from
  end
  subroutine test
    type(t) x
    !ERROR: Multiple specific procedures for the generic ASSIGNMENT(=) match operand types TYPE(t) and INTEGER(4)
    x = 5
  end
end
