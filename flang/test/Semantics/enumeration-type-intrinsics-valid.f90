! RUN: %flang_fc1 -fsyntax-only -fenumeration-type -pedantic %s
! NEXT/PREVIOUS calls that must compile cleanly (F2023 16.9.151, 16.9.164).

module enum_intrinsics_valid_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type
end module

subroutine test_nonconstant_a(c, arr)
  use enum_intrinsics_valid_mod
  type(color), intent(in) :: c, arr(3)
  type(color) :: r, rarr(3)
  r = next(c)
  r = previous(c)
  rarr = next(arr)
  rarr = previous(arr)
end subroutine

subroutine test_stat(c, arr)
  use enum_intrinsics_valid_mod
  type(color), intent(in) :: c, arr(3)
  type(color) :: r, rarr(3)
  integer :: st, starr(3)
  r = next(c, stat=st)
  r = previous(c, stat=st)
  r = next(blue, stat=st)
  r = previous(red, stat=st)
  r = next(huge(red), stat=st)
  rarr = next(arr, stat=starr)
  rarr = previous(arr, stat=starr)
  rarr = next(c, stat=starr)
  rarr = previous(c, stat=starr)
end subroutine

! A boundary without STAT= outside a constant expression is a run-time error
! termination, not a compile-time error.
subroutine test_boundary_nonconstant_context()
  use enum_intrinsics_valid_mod
  type(color) :: r, rarr(2)
  r = next(blue)
  r = previous(red)
  rarr = next([green, blue])
  rarr = previous([red, green])
end subroutine

subroutine test_stat_kinds(c, arr)
  use enum_intrinsics_valid_mod
  type(color), intent(in) :: c, arr(3)
  type(color) :: r, rarr(3)
  integer(2) :: s2, s2arr(3)
  integer(8) :: s8, s8arr(3)
  r = next(c, stat=s2)
  r = previous(c, stat=s8)
  rarr = next(arr, stat=s2arr)
  rarr = previous(arr, stat=s8arr)
end subroutine

subroutine test_keyword_order(c)
  use enum_intrinsics_valid_mod
  type(color), intent(in) :: c
  type(color) :: r
  integer :: st
  r = next(stat=st, a=c)
  r = previous(stat=st, a=c)
end subroutine
