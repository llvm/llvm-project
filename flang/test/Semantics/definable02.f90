! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

! Ensure that FINAL subroutine can be called for array with vector-valued
! subscript.

module m
  type t1
   contains
    final :: f1
  end type
  type t2
   contains
    final :: f2
  end type
  type t3
   contains
    final :: f3
  end type
 contains
  subroutine f1(x)
    type(t1), intent(in out) :: x(:)
  end subroutine
  subroutine f2(x)
    type(t2), intent(in out) :: x(..)
  end subroutine
  impure elemental subroutine f3(x)
    type(t3), intent(in out) :: x
  end subroutine
end module

program test
  use m
  type(t1) x1(1)
  type(t2) x2(1)
  type(t3) x3(1)
  x1(:) = [t1()] ! ok
  x2(:) = [t2()] ! ok
  x3(:) = [t3()] ! ok
  !PORTABILITY: Variable 'x1([INTEGER(8)::1_8])' has a vector subscript and will be finalized by non-elemental subroutine 'f1'
  x1([1]) = [t1()]
  !PORTABILITY: Variable 'x2([INTEGER(8)::1_8])' has a vector subscript and will be finalized by non-elemental subroutine 'f2'
  x2([1]) = [t2()]
  x3([1]) = [t3()] ! ok
end
