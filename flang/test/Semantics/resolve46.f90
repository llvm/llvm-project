! RUN: %python %S/test_errors.py %s %flang_fc1
! C1030 - pointers to intrinsic procedures
program main
  intrinsic :: cos ! a specific & generic intrinsic name
  intrinsic :: alog10 ! a specific intrinsic name, not generic
  intrinsic :: null ! a weird special case
  intrinsic :: bessel_j0 ! generic intrinsic, not specific
  intrinsic :: amin0
  !ERROR: 'haltandcatchfire' is not a known intrinsic procedure
  intrinsic :: haltandcatchfire
  procedure(sin), pointer :: p
  p => alog ! valid use of an unrestricted specific intrinsic
  p => alog10 ! ditto, but already declared intrinsic
  p => cos ! ditto, but also generic
  p => tan ! a generic & an unrestricted specific, not already declared
  !ERROR: Procedure pointer 'p' associated with incompatible procedure designator 'amin0'
  p => amin0
  !ERROR: Procedure pointer 'p' associated with incompatible procedure designator 'amin1'
  p => amin1
  !ERROR: 'bessel_j0' is not a specific intrinsic procedure
  p => bessel_j0
end program main
