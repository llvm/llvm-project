! RUN: %python %S/test_errors.py %s %flang_fc1
! Test 8.5.10 & 8.5.18 constraints on dummy argument declarations

module m

  type :: hasCoarray
    real, allocatable :: a(:)[:]
  end type
  type, extends(hasCoarray) :: extendsHasCoarray
  end type
  type :: hasCoarray2
    type(hasCoarray) :: x
  end type
  type, extends(hasCoarray2) :: extendsHasCoarray2
  end type

  real, allocatable :: coarray(:)[:]

 contains

  subroutine s01a(x)
    real, allocatable, intent(out) :: x(:)
  end subroutine
  subroutine s01b ! C846 - can only be caught at a call via explicit interface
    !ERROR: ALLOCATABLE coarray 'coarray' may not be associated with INTENT(OUT) dummy argument 'x='
    !ERROR: ALLOCATABLE dummy argument 'x=' has corank 0 but actual argument has corank 1
    call s01a(coarray)
  end subroutine

  subroutine s02(x) ! C846
    !ERROR: An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray
    type(hasCoarray), intent(out) :: x
  end subroutine

  subroutine s03(x) ! C846
    !ERROR: An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray
    type(extendsHasCoarray), intent(out) :: x
  end subroutine

  subroutine s04(x) ! C846
    !ERROR: An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray
    type(hasCoarray2), intent(out) :: x
  end subroutine

  subroutine s05(x) ! C846
    !ERROR: An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray
    type(extendsHasCoarray2), intent(out) :: x
  end subroutine

end module

subroutine s06(x) ! C847
  use ISO_FORTRAN_ENV, only: lock_type
  !ERROR: An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE
  type(lock_type), intent(out) :: x
end subroutine

subroutine s07(x) ! C847
  use ISO_FORTRAN_ENV, only: event_type
  !ERROR: An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE
  type(event_type), intent(out) :: x
end subroutine
