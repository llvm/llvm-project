! Test that NEXT/PREVIOUS with STAT= inside FORALL hits the TODO diagnostic.
! RUN: %not_todo_cmd %flang_fc1 -fenumeration-type -emit-hlfir -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: NEXT/PREVIOUS with STAT= inside FORALL
module enum_forall_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

subroutine test_forall_next_stat(a, n, st)
  use enum_forall_mod
  type(color), intent(in) :: a(3)
  type(color), intent(out) :: n(3)
  integer, intent(out) :: st(3)
  integer :: i
  forall (i = 1:3)
    n(i) = next(a(i), stat=st(i))
  end forall
end subroutine
