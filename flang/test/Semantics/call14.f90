! RUN: %S/test_errors.sh %s %t %f18
! Test 8.5.18 constraints on the VALUE attribute

module m
  type :: hasCoarray
    real, allocatable :: coarray[:]
  end type
 contains
  !ERROR: VALUE attribute may apply only to a dummy data object
  subroutine C863(notData,assumedSize,coarray,coarrayComponent)
    external :: notData
    !ERROR: VALUE attribute may apply only to a dummy argument
    real, value :: notADummy
    value :: notData
    !ERROR: VALUE attribute may not apply to an assumed-size array
    real, value :: assumedSize(10,*)
    !ERROR: VALUE attribute may not apply to a coarray
    real, value :: coarray[*]
    !ERROR: VALUE attribute may not apply to a type with a coarray ultimate component
    type(hasCoarray), value :: coarrayComponent
  end subroutine
  subroutine C864(allocatable, inout, out, pointer, volatile)
    !ERROR: VALUE attribute may not apply to an ALLOCATABLE
    real, value, allocatable :: allocatable
    !ERROR: VALUE attribute may not apply to an INTENT(IN OUT) argument
    real, value, intent(in out) :: inout
    !ERROR: VALUE attribute may not apply to an INTENT(OUT) argument
    real, value, intent(out) :: out
    !ERROR: VALUE attribute may not apply to a POINTER
    real, value, pointer :: pointer
    !ERROR: VALUE attribute may not apply to a VOLATILE
    real, value, volatile :: volatile
  end subroutine
  subroutine C865(optional) bind(c)
    !ERROR: VALUE attribute may not apply to an OPTIONAL in a BIND(C) procedure
    real, value, optional :: optional
  end subroutine
end module
