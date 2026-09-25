! RUN: not %flang_fc1 -fsyntax-only -fenumeration-type -pedantic %s 2>&1 | FileCheck %s
! Test intrinsics HUGE, NEXT, PREVIOUS, INT for enumeration types (F2023 7.6.2).

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

  ! NEXT(a) with a non-constant argument now lowers at run time.
  c = red
  nc = next(c)

  ! NEXT with constants
  nc = next(red)
  nc = next(green)

  ! NEXT with STAT= (now supported).
  nc = next(c, stat=istat)
  nc = next(blue, stat=istat)
end subroutine

subroutine test_previous()
  use enum_intrinsics_mod
  type(color) :: c, pc
  integer :: istat

  ! PREVIOUS(a) with a non-constant argument now lowers at run time.
  c = blue
  pc = previous(c)

  ! PREVIOUS with constants
  pc = previous(blue)
  pc = previous(green)

  ! PREVIOUS with STAT= (now supported).
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

subroutine test_int_bad_kind()
  use enum_intrinsics_mod
  integer :: i
  ! INT of an enumeration argument with an unsupported KIND value is rejected.
  !CHECK: error: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
  i = int(red, kind=3)
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
  ! NEXT at a boundary WITH STAT= is valid: the boundary is reported at run
  ! time via STAT=, so nothing is diagnosed at compile time.
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
  ! PREVIOUS at a boundary WITH STAT= is valid (boundary reported at run time).
  pc = previous(red, stat=istat)
end subroutine

subroutine test_next_boundary()
  use enum_intrinsics_mod
  type(color) :: nc
  ! NEXT at the last enumerator without STAT in a non-constant context is a
  ! run-time error termination; it is deferred to lowering, so nothing is
  ! diagnosed here at compile time.
  nc = next(blue)
end subroutine

subroutine test_previous_boundary()
  use enum_intrinsics_mod
  type(color) :: pc
  ! PREVIOUS at the first enumerator without STAT in a non-constant context is
  ! deferred to run time; nothing is diagnosed at compile time.
  pc = previous(red)
end subroutine

subroutine test_next_previous_array_boundary()
  use enum_intrinsics_mod
  type(color) :: nc(2), pc(2)
  ! NEXT/PREVIOUS are elemental: an array with a boundary element without STAT=
  ! in a non-constant context is deferred to run-time error termination, so it
  ! is not diagnosed at compile time.
  nc = next([green, blue])
  pc = previous([red, green])
end subroutine

subroutine test_next_previous_boundary_constant()
  use enum_intrinsics_mod
  ! A required-constant boundary case cannot be deferred to run time, so it is
  ! diagnosed as out of range at compile time.
  !CHECK: error: NEXT() of the last enumerator is out of range
  logical, parameter :: nb = next(blue) == green
  !CHECK: error: PREVIOUS() of the first enumerator is out of range
  logical, parameter :: pb = previous(red) == green
end subroutine

subroutine test_next_previous_array_boundary_constant()
  use enum_intrinsics_mod
  ! Elemental boundary hit in a required-constant array context is likewise
  ! diagnosed at compile time.
  !CHECK: error: NEXT() of the last enumerator is out of range
  type(color), parameter :: nbad(2) = next([green, blue])
  !CHECK: error: PREVIOUS() of the first enumerator is out of range
  type(color), parameter :: pbad(2) = previous([red, green])
end subroutine

subroutine test_huge_real_still_works()
  ! Non-enumeration HUGE still works normally
  real :: r
  integer :: i
  r = huge(r)
  i = huge(i)
end subroutine

subroutine test_next_previous_keyword_order()
  use enum_intrinsics_mod
  type(color) :: nc
  integer :: istat
  ! The enum argument passed by keyword AFTER a non-enum keyword (STAT=) must
  ! still be recognized as the enumeration call; these now compile cleanly.
  nc = next(stat=istat, a=red)
  nc = previous(stat=istat, a=blue)
end subroutine

subroutine test_next_previous_stat_nonconformant()
  use enum_intrinsics_mod
  type(color) :: arr(3), nc(3), pc(3)
  integer :: stat2(2)
  ! NEXT/PREVIOUS are elemental with an INTENT(OUT) STAT=, so a STAT= array
  ! must conform with A; a differently shaped STAT= is caught by the general
  ! elemental-conformance check on the resolved call.
  !CHECK: error: Dimension 1 of actual argument (arr) corresponding to dummy argument #1 ('a') has extent 3, but actual argument (stat2) corresponding to dummy argument #2 ('stat') has extent 2
  nc = next(arr, stat=stat2)
  !CHECK: error: Dimension 1 of actual argument (arr) corresponding to dummy argument #1 ('a') has extent 3, but actual argument (stat2) corresponding to dummy argument #2 ('stat') has extent 2
  pc = previous(arr, stat=stat2)
end subroutine
