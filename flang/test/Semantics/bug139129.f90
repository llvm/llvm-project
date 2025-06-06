!RUN: %flang_fc1 -fsyntax-only %s
module m
  type t
   contains
    procedure asst
    generic :: assignment(=) => asst
  end type
 contains
  pure subroutine asst(lhs, rhs)
    class(t), intent(in out) :: lhs
    class(t), intent(in) :: rhs
  end
  pure subroutine test(x, y)
    class(t), intent(in out) :: x, y
    x = y ! spurious definability error
  end
end
