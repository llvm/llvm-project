! RUN: %python %S/test_errors.py %s %flang_fc1
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

subroutine test_intrinsic_next_declaration()
  ! Explicitly declaring NEXT intrinsic here (with the enumeration-type
  ! feature disabled) must produce only the single expected diagnostic below.
  ! Before the DeclareIntrinsic fix, the name was still incorrectly flagged
  ! as an intrinsic function under the hood, risking a second, bogus
  ! diagnostic when it was later referenced as a call below.
  !ERROR: 'next' is not a known intrinsic procedure
  intrinsic :: next
  integer :: i
  i = next(5)
end subroutine
