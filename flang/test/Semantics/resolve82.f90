! RUN: %python %S/test_errors.py %s %flang_fc1
! C815 An entity shall not be explicitly given any attribute more than once in 
! a scoping unit.
!
! R1512 procedure-declaration-stmt ->
!         PROCEDURE ( [proc-interface] ) [[, proc-attr-spec]... ::]
!         proc-decl-list
!  proc-attr-spec values are:
!    PUBLIC, PRIVATE, BIND(C), INTENT (intent-spec), OPTIONAL, POINTER, 
!    PROTECTED, SAVE
module m
  abstract interface
    real function procFunc()
    end function procFunc
  end interface

  !WARNING: Attribute 'PUBLIC' cannot be used more than once
  procedure(procFunc), public, pointer, public :: proc1
  !WARNING: Attribute 'PRIVATE' cannot be used more than once
  procedure(procFunc), private, pointer, private :: proc2
  !WARNING: Attribute 'BIND(C)' cannot be used more than once
  procedure(procFunc), bind(c), pointer, bind(c) :: proc3
  !WARNING: Attribute 'PROTECTED' cannot be used more than once
  procedure(procFunc), protected, pointer, protected :: proc4

contains

    subroutine testProcDecl(arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11)
      !WARNING: Attribute 'INTENT(IN)' cannot be used more than once
      procedure(procFunc), intent(in), pointer, intent(in) :: arg4
      !WARNING: Attribute 'INTENT(OUT)' cannot be used more than once
      procedure(procFunc), intent(out), pointer, intent(out) :: arg5
      !WARNING: Attribute 'INTENT(INOUT)' cannot be used more than once
      procedure(procFunc), intent(inout), pointer, intent(inout) :: arg6
      !ERROR: Attributes 'INTENT(INOUT)' and 'INTENT(OUT)' conflict with each other
      procedure(procFunc), intent(inout), pointer, intent(out) :: arg7
      !ERROR: Attributes 'INTENT(INOUT)' and 'INTENT(OUT)' conflict with each other
      procedure(procFunc), intent(out), pointer, intent(inout) :: arg8
      !WARNING: Attribute 'OPTIONAL' cannot be used more than once
      procedure(procFunc), optional, pointer, optional :: arg9
      !WARNING: Attribute 'POINTER' cannot be used more than once
      procedure(procFunc), pointer, optional, pointer :: arg10
      !WARNING: Attribute 'SAVE' cannot be used more than once
      procedure(procFunc), save, pointer, save :: localProc
    end subroutine testProcDecl

end module m
