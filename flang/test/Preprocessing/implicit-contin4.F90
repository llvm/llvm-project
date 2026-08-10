! RUN: %flang -E %s | FileCheck %s
! Macro definitions with unbalanced parentheses should not affect
! implicit continuations.
subroutine foo(a, d)
  implicit none
  integer :: a
  integer :: d

#define sub(x, y) foo2(x, y)
#define bar )

   call sub(1,
     2)
end subroutine foo

!CHECK: call foo2(1, 2)
!CHECK: end subroutine foo
