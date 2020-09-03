! RUN: %S/test_errors.sh %s %t %f18
module m
! C755 The same proc-component-attr-spec shall not appear more than once in a 
! given proc-component-def-stmt.
! C759 PASS and NOPASS shall not both appear in the same 
! proc-component-attr-spec-list.
!
! R741 proc-component-def-stmt ->
!        PROCEDURE ( [proc-interface] ) , proc-component-attr-spec-list
!          :: proc-decl-list
!  proc-component-attr-spec values are:
!    PUBLIC, PRIVATE, NOPASS, PASS, POINTER

  type :: procComponentType
    !WARNING: Attribute 'PUBLIC' cannot be used more than once
    procedure(publicProc), public, pointer, public :: publicField
    !WARNING: Attribute 'PRIVATE' cannot be used more than once
    procedure(privateProc), private, pointer, private :: privateField
    !WARNING: Attribute 'NOPASS' cannot be used more than once
    procedure(nopassProc), nopass, pointer, nopass :: noPassField
    !WARNING: Attribute 'PASS' cannot be used more than once
    procedure(passProc), pass, pointer, pass :: passField
    !ERROR: Attributes 'PASS' and 'NOPASS' conflict with each other
    procedure(passNopassProc), pass, pointer, nopass :: passNopassField
    !WARNING: Attribute 'POINTER' cannot be used more than once
    procedure(pointerProc), pointer, public, pointer :: pointerField
    !ERROR: Procedure component 'nonpointerfield' must have POINTER attribute
    procedure(publicProc), public :: nonpointerField
  contains
    procedure :: noPassProc
    procedure :: passProc
    procedure :: passNopassProc
    procedure :: publicProc
    procedure :: privateProc
  end type procComponentType

contains
    subroutine publicProc(arg)
      class(procComponentType) :: arg
    end
    subroutine privateProc(arg)
      class(procComponentType) :: arg
    end
    subroutine noPassProc(arg)
      class(procComponentType) :: arg
    end
    subroutine passProc(arg)
      class(procComponentType) :: arg
    end
    subroutine passNopassProc(arg)
      class(procComponentType) :: arg
    end
    subroutine pointerProc(arg)
      class(procComponentType) :: arg
    end
end module m
