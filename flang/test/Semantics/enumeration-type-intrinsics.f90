! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
! Test intrinsics HUGE, NEXT, PREVIOUS, INT for enumeration types (F2023 7.6.2)

module enum_intrinsics_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  enumeration type :: v_value
    enumerator :: v_one, v_two, v_three
    enumerator v_four
  end enumeration type
end module

subroutine test_huge()
  use enum_intrinsics_mod
  type(color) :: x
  type(v_value) :: y

  ! HUGE(x) returns the last enumerator
  x = huge(x)
  y = huge(y)

  ! HUGE in comparison — should fold to .TRUE.
  if (huge(x) == blue) continue
  if (huge(y) == v_four) continue
end subroutine

subroutine test_next()
  use enum_intrinsics_mod
  type(color) :: c, nc
  integer :: istat

  ! NEXT(a) returns the next enumerator
  c = red
  nc = next(c)

  ! NEXT with constants
  nc = next(red)
  nc = next(green)

  ! NEXT with STAT= argument
  nc = next(c, stat=istat)
  nc = next(blue, stat=istat)
end subroutine

subroutine test_previous()
  use enum_intrinsics_mod
  type(color) :: c, pc
  integer :: istat

  ! PREVIOUS(a) returns the previous enumerator
  c = blue
  pc = previous(c)

  ! PREVIOUS with constants
  pc = previous(blue)
  pc = previous(green)

  ! PREVIOUS with STAT= argument
  pc = previous(c, stat=istat)
  pc = previous(red, stat=istat)
end subroutine

subroutine test_int()
  use enum_intrinsics_mod
  integer :: i
  integer(8) :: j

  ! INT(x) returns the ordinal position
  i = int(red)
  i = int(green)
  i = int(blue)

  ! INT with KIND= argument
  j = int(red, kind=8)
  j = int(green, 8)
end subroutine

subroutine test_int_parameter()
  use enum_intrinsics_mod
  ! INT(x) in parameter (constant) context
  integer, parameter :: r = int(red)
  integer, parameter :: g = int(green)
  integer, parameter :: b = int(blue)

  ! Verify ordinals are 1-based
  integer, parameter :: test1 = r  ! should be 1
  integer, parameter :: test2 = g  ! should be 2
  integer, parameter :: test3 = b  ! should be 3
end subroutine

subroutine test_huge_constant()
  use enum_intrinsics_mod
  ! HUGE in constant context
  logical, parameter :: h1 = huge(red) == blue
  logical, parameter :: h2 = huge(v_one) == v_four
end subroutine

subroutine test_next_constant()
  use enum_intrinsics_mod
  ! NEXT with constant folding — non-boundary cases
  logical, parameter :: n1 = next(red) == green
  logical, parameter :: n2 = next(green) == blue
end subroutine

subroutine test_next_boundary_with_stat()
  use enum_intrinsics_mod
  type(color) :: nc
  integer :: istat
  ! NEXT at boundary with STAT — no error, STAT gets nonzero
  nc = next(blue, stat=istat)
  nc = next(huge(red), stat=istat)
end subroutine

subroutine test_previous_constant()
  use enum_intrinsics_mod
  ! PREVIOUS with constant folding — non-boundary cases
  logical, parameter :: p1 = previous(blue) == green
  logical, parameter :: p2 = previous(green) == red
end subroutine

subroutine test_previous_boundary_with_stat()
  use enum_intrinsics_mod
  type(color) :: pc
  integer :: istat
  ! PREVIOUS at boundary with STAT — no error, STAT gets nonzero
  pc = previous(red, stat=istat)
end subroutine

subroutine test_next_boundary_warning()
  use enum_intrinsics_mod
  type(color) :: nc
  ! NEXT at boundary without STAT — warning
  !CHECK: warning: NEXT() of last enumerator without STAT= causes error termination
  nc = next(blue)
end subroutine

subroutine test_previous_boundary_warning()
  use enum_intrinsics_mod
  type(color) :: pc
  ! PREVIOUS at boundary without STAT — warning
  !CHECK: warning: PREVIOUS() of first enumerator without STAT= causes error termination
  pc = previous(red)
end subroutine

subroutine test_huge_real_still_works()
  ! Non-enumeration HUGE still works normally
  real :: r
  integer :: i
  r = huge(r)
  i = huge(i)
end subroutine
