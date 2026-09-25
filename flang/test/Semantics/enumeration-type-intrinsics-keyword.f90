! RUN: %flang_fc1 -fsyntax-only -fenumeration-type %s
! Verify keyword-order argument binding for enumeration-type intrinsics.
! INT(KIND=..., A=...) must recognize the enumeration argument even when it is
! not the first positional actual.  This is a regression test: previously the
! enum dispatch inspected only the first positional argument, so a keyword call
! that placed KIND= before A= reported "Actual argument for 'a=' has bad type".
! A clean compile (exit 0) confirms the enum path is taken for the keyword form.

module enum_keyword_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

subroutine test_int_keyword_order()
  use enum_keyword_mod
  integer :: i
  integer(8) :: j

  ! Enum argument bound by keyword A=, KIND= appears first.
  j = int(kind=8, a=red)
  i = int(a=green, kind=4)
  j = int(a=blue, kind=8)

  ! Positional forms continue to work.
  i = int(red)
  j = int(green, 8)
end subroutine

subroutine test_huge_keyword()
  use enum_keyword_mod
  type(color) :: x

  ! HUGE bound by keyword X=.
  x = huge(x=x)
end subroutine

subroutine test_next_previous_keyword()
  use enum_keyword_mod
  type(color) :: c

  ! Enum argument bound by keyword A= (first and only dummy).
  c = next(a=red)
  c = previous(a=blue)
end subroutine
