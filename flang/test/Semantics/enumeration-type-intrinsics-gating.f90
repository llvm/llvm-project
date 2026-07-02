! RUN: %flang_fc1 -fsyntax-only %s
! Without -fenumeration-type, NEXT and PREVIOUS are not reserved intrinsic
! names, so a pre-F2023 program may use them as implicit external procedures.
! This exercises the enumeration-type feature gating in resolve-names.cpp and
! verifies the reference no longer triggers an internal compiler error.
! NOTE: This gating is TEMPORARY and is removed once the enumeration-type
!       feature is fully implemented.

program p
  integer :: i
  real :: r
  i = next(5)
  r = previous(3)
end program
