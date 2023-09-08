! RUN: %python %S/test_errors.py %s %flang_fc1
! C1030 - assignment of pointers to intrinsic procedures
! C1515 - interface definition for procedure pointers
! C1519 - initialization of pointers to intrinsic procedures
program main
  intrinsic :: cos ! a specific & generic intrinsic name
  intrinsic :: alog10 ! a specific intrinsic name, not generic
  intrinsic :: null ! a weird special case
  intrinsic :: bessel_j0 ! generic intrinsic, not specific
  intrinsic :: amin0
  intrinsic :: mod
  intrinsic :: llt
  !ERROR: 'haltandcatchfire' is not a known intrinsic procedure
  intrinsic :: haltandcatchfire

  abstract interface
     logical function chrcmp(a,b)
       character(*), intent(in) :: a
       character(*), intent(in) :: b
     end function chrcmp
  end interface

  !PORTABILITY: Procedure pointer 'p' should not have an ELEMENTAL intrinsic as its interface
  procedure(sin), pointer :: p => cos
  !ERROR: Intrinsic procedure 'amin0' is not an unrestricted specific intrinsic permitted for use as the definition of the interface to procedure pointer 'q'
  procedure(amin0), pointer :: q
  !ERROR: Intrinsic procedure 'bessel_j0' is not an unrestricted specific intrinsic permitted for use as the definition of the interface to procedure pointer 'r'
  procedure(bessel_j0), pointer :: r
  !ERROR: Intrinsic procedure 'llt' is not an unrestricted specific intrinsic permitted for use as the initializer for procedure pointer 's'
  procedure(chrcmp), pointer :: s => llt
  !ERROR: Intrinsic procedure 'bessel_j0' is not an unrestricted specific intrinsic permitted for use as the initializer for procedure pointer 't'
  !PORTABILITY: Procedure pointer 't' should not have an ELEMENTAL intrinsic as its interface
  procedure(cos), pointer :: t => bessel_j0
  procedure(chrcmp), pointer :: u
  p => alog ! valid use of an unrestricted specific intrinsic
  p => alog10 ! ditto, but already declared intrinsic
  p => cos ! ditto, but also generic
  p => tan ! a generic & an unrestricted specific, not already declared
  !ERROR: Function pointer 'p' associated with incompatible function designator 'mod': function results have distinct types: REAL(4) vs INTEGER(4)
  p => mod
  !ERROR: Function pointer 'p' associated with incompatible function designator 'index': function results have distinct types: REAL(4) vs INTEGER(4)
  p => index
  !ERROR: 'bessel_j0' is not an unrestricted specific intrinsic procedure
  p => bessel_j0
  !ERROR: 'llt' is not an unrestricted specific intrinsic procedure
  u => llt
end program main
