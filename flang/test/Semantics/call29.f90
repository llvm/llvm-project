! RUN: %python %S/test_errors.py %s %flang_fc1

module m
  type t1
    integer, allocatable :: a(:)
  end type
  type t2
    integer :: n = 123
  end type
  type t3
   contains
    final :: t3final
  end type
  type t4
    type(t1) :: c1
    type(t2) :: c2
    type(t3) :: c3
  end type
  type t5
  end type
 contains
  elemental subroutine t3final(x)
    type(t3), intent(in) :: x
  end subroutine
  subroutine test1(x1,x2,x3,x4,x5)
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not have a derived type with any default component initialization
    type(t1), intent(out) :: x1(*)
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not have a derived type with any default component initialization
    type(t2), intent(out) :: x2(*)
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not be finalizable
    type(t3), intent(out) :: x3(*)
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not have a derived type with any default component initialization
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not be finalizable
    type(t4), intent(out) :: x4(*)
    !ERROR: An INTENT(OUT) assumed-size dummy argument array may not be polymorphic
    class(t5), intent(out) :: x5(*)
  end subroutine
end module
