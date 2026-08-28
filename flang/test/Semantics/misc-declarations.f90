! RUN: %python %S/test_errors.py %s %flang_fc1
! Miscellaneous constraint and requirement checking on declarations:
! - 8.5.6.2 & 8.5.6.3 constraints on coarrays
! - 8.5.19 constraints on the VOLATILE attribute

module m
  !ERROR: 'mustbedeferred' is an ALLOCATABLE coarray and must have a deferred coshape
  real, allocatable :: mustBeDeferred[*]  ! C827
  !ERROR: 'mustbeexplicit' is a non-ALLOCATABLE coarray and must have an explicit coshape
  real :: mustBeExplicit[:]  ! C828
  type :: hasCoarray
    real, allocatable :: coarray[:]
  end type
  real :: coarray[*]
  type(hasCoarray) :: coarrayComponent
 contains
  !ERROR: VOLATILE attribute may not apply to an INTENT(IN) argument
  subroutine C866(x)
    intent(in) :: x
    volatile :: x
    !ERROR: VOLATILE attribute may apply only to a variable
    volatile :: notData
    external :: notData
  end subroutine
  subroutine C867
    !ERROR: VOLATILE attribute may not apply to a coarray accessed by USE or host association
    volatile :: coarray
    !ERROR: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
    volatile :: coarrayComponent
  end subroutine
  subroutine C868(coarray,coarrayComponent)
    real :: coarray[*]
    type(hasCoarray) :: coarrayComponent
    block
      !ERROR: VOLATILE attribute may not apply to a coarray accessed by USE or host association
      volatile :: coarray
      !ERROR: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
      volatile :: coarrayComponent
    end block
  end subroutine
  subroutine C839(x)
    !ERROR: Coarray 'x' may not be an assumed-rank array
    real, intent(in) :: x(..)[*]
  end
end module

! Submodule host-association: same C867/C868-equivalent constraints apply
! when VOLATILE names a host variable from the ancestor module.
module m2
  real :: smCoarray[*]
  type :: hasCoarray2
    real, allocatable :: coarray[:]
  end type
  type(hasCoarray2) :: smCoarrayComp
  interface
    module subroutine smProc()
    end subroutine
  end interface
end module
submodule(m2) sm2
  !ERROR: VOLATILE attribute may not apply to a coarray accessed by USE or host association
  volatile :: smCoarray
  !ERROR: VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association
  volatile :: smCoarrayComp
  volatile :: smProc
contains
  !ERROR: VOLATILE attribute may apply only to a variable
  module subroutine smProc()
  end subroutine
end submodule
