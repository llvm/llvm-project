! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Ensures that parentheses are preserved with derived types
module m
  type :: t
    integer :: n
  end type
 contains
  subroutine sub(x)
    type(t), intent(in) :: x
  end subroutine
  function f(m)
    type(t), pointer :: f
    integer, intent(in) :: m
    type(t), save, target :: res
    res%n = m
    f => res
  end function
  subroutine test
    type(t) :: x
    x = t(1)
    !CHECK: CALL sub(t(n=1_4))
    call sub(t(1))
    !CHECK: CALL sub((t(n=1_4)))
    call sub((t(1)))
    !CHECK: CALL sub(x)
    call sub(x)
    !CHECK: CALL sub((x))
    call sub((x))
    !CHECK: CALL sub(f(2_4))
    call sub(f(2))
    !CHECK: CALL sub((f(2_4)))
    call sub((f(2)))
  end subroutine
end module
