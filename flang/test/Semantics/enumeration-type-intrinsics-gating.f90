! RUN: %python %S/test_errors.py %s %flang_fc1
! NEXT and PREVIOUS are F2023 intrinsic functions.  A pre-F2023 program that
! references an external function with either name must declare it EXTERNAL
! or give it an explicit interface.

program p
  integer :: i
  real :: r
  !ERROR: Argument of NEXT() must be of enumeration type
  i = next(5)
  !ERROR: Argument of PREVIOUS() must be of enumeration type
  r = previous(3)
end program

subroutine test_external()
  external :: next, previous
  integer :: next
  real :: previous
  integer :: i
  real :: r
  i = next(5)
  r = previous(3)
end subroutine

subroutine test_intrinsic_declaration()
  intrinsic :: next, previous
  integer :: i
  !ERROR: Argument of NEXT() must be of enumeration type
  i = next(5)
end subroutine

! The intrinsics are functions, so a CALL still refers to an external subroutine.
subroutine test_call()
  call next(1)
  call previous(1)
end subroutine
