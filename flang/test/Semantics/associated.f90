! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Tests for the ASSOCIATED() and NULL() intrinsics
subroutine assoc()

  abstract interface
    subroutine subrInt(i)
      integer :: i
    end subroutine subrInt

    integer function abstractIntFunc(x)
      integer, intent(in) :: x
    end function
  end interface

  contains
  integer function intFunc(x)
    integer, intent(in) :: x
    intFunc = x
  end function

  real function realFunc(x)
    real, intent(in) :: x
    realFunc = x
  end function

  pure integer function pureFunc()
    pureFunc = 343
  end function pureFunc

  elemental integer function elementalFunc()
    elementalFunc = 343
  end function elementalFunc

  subroutine subr(i)
    integer :: i
  end subroutine subr

  subroutine test()
    integer :: intVar
    integer, target :: targetIntVar1
    integer(kind=2), target :: targetIntVar2
    real, target :: targetRealVar
    integer, pointer :: intPointerVar1
    integer, pointer :: intPointerVar2
    integer, allocatable :: intAllocVar
    procedure(intFunc) :: intProc
    procedure(intFunc), pointer :: intprocPointer1
    procedure(intFunc), pointer :: intprocPointer2
    procedure(realFunc) :: realProc
    procedure(realFunc), pointer :: realprocPointer1
    procedure(pureFunc), pointer :: pureFuncPointer
    procedure(elementalFunc) :: elementalProc
    external :: externalProc
    procedure(subrInt) :: subProc
    procedure(subrInt), pointer :: subProcPointer
    procedure(), pointer :: implicitProcPointer
    logical :: lVar

    !ERROR: missing mandatory 'pointer=' argument
    lVar = associated()
    !ERROR: MOLD= argument to NULL() must be a pointer or allocatable
    lVar = associated(null(intVar))
    lVar = associated(null(intAllocVar)) !OK
    lVar = associated(null()) !OK
    lVar = associated(null(intPointerVar1)) !OK
    lVar = associated(null(), null()) !OK
    lVar = associated(intPointerVar1, null(intPointerVar2)) !OK
    lVar = associated(intPointerVar1, null()) !OK
    lVar = associated(null(), null(intPointerVar1)) !OK
    lVar = associated(null(intPointerVar1), null()) !OK
    !ERROR: POINTER= argument of ASSOCIATED() must be a POINTER
    lVar = associated(intVar)
    !ERROR: POINTER= argument of ASSOCIATED() must be a POINTER
    lVar = associated(intVar, intVar)
    !ERROR: POINTER= argument of ASSOCIATED() must be a POINTER
    lVar = associated(intAllocVar)
    !ERROR: Arguments of ASSOCIATED() must be a POINTER and an optional valid target
    lVar = associated(intPointerVar1, targetRealVar)
    lVar = associated(intPointerVar1, targetIntVar1) !OK
    !ERROR: Arguments of ASSOCIATED() must be a POINTER and an optional valid target
    lVar = associated(intPointerVar1, targetIntVar2)
    lVar = associated(intPointerVar1) !OK
    lVar = associated(intPointerVar1, intPointerVar2) !OK
    !ERROR: In assignment to object pointer 'intpointervar1', the target 'intvar' is not an object with POINTER or TARGET attributes
    intPointerVar1 => intVar
    !ERROR: TARGET= argument 'intvar' must have either the POINTER or the TARGET attribute
    lVar = associated(intPointerVar1, intVar)

    ! Procedure pointer tests
    intprocPointer1 => intProc !OK
    lVar = associated(intprocPointer1, intProc) !OK
    intprocPointer1 => intProcPointer2 !OK
    lVar = associated(intprocPointer1, intProcPointer2) !OK
    intProcPointer1  => null(intProcPointer2) ! ok
    lvar = associated(intProcPointer1, null(intProcPointer2)) ! ok
    intProcPointer1 => null() ! ok
    lvar = associated(intProcPointer1, null()) ! ok
    intProcPointer1 => intProcPointer2 ! ok
    lvar = associated(intProcPointer1, intProcPointer2) ! ok
    intProcPointer1 => null(intProcPointer2) ! ok
    lvar = associated(intProcPointer1, null(intProcPointer2)) ! ok
    intProcPointer1 =>null() ! ok
    lvar = associated(intProcPointer1, null()) ! ok
    intPointerVar1 => null(intPointerVar1) ! ok
    lvar = associated (intPointerVar1, null(intPointerVar1)) ! ok

    !ERROR: In assignment to procedure pointer 'intprocpointer1', the target is not a procedure or procedure pointer
    intprocPointer1 => intVar
    !ERROR: POINTER= argument 'intprocpointer1' is a procedure pointer but the TARGET= argument 'intvar' is not a procedure or procedure pointer
    lVar = associated(intprocPointer1, intVar)
    !ERROR: Procedure pointer 'intprocpointer1' associated with incompatible procedure designator 'elementalproc'
    intProcPointer1 => elementalProc
    !ERROR: Procedure pointer 'intprocpointer1' associated with incompatible procedure designator 'elementalproc'
    lvar = associated(intProcPointer1, elementalProc)
    !ERROR: POINTER= argument 'intpointervar1' is an object pointer but the TARGET= argument 'intfunc' is a procedure designator
    lvar = associated (intPointerVar1, intFunc)
    !ERROR: In assignment to object pointer 'intpointervar1', the target 'intfunc' is a procedure designator
    intPointerVar1 => intFunc
    !ERROR: In assignment to procedure pointer 'intprocpointer1', the target is not a procedure or procedure pointer
    intProcPointer1 => targetIntVar1
    !ERROR: POINTER= argument 'intprocpointer1' is a procedure pointer but the TARGET= argument 'targetintvar1' is not a procedure or procedure pointer
    lvar = associated (intProcPointer1, targetIntVar1)
    !ERROR: Procedure pointer 'intprocpointer1' associated with result of reference to function 'null' that is an incompatible procedure pointer
    intProcPointer1 => null(mold=realProcPointer1)
    !ERROR: Procedure pointer 'intprocpointer1' associated with result of reference to function 'null()' that is an incompatible procedure pointer
    lvar = associated(intProcPointer1, null(mold=realProcPointer1))
    !ERROR: PURE procedure pointer 'purefuncpointer' may not be associated with non-PURE procedure designator 'intproc'
    pureFuncPointer => intProc
    !ERROR: PURE procedure pointer 'purefuncpointer' may not be associated with non-PURE procedure designator 'intproc'
    lvar = associated(pureFuncPointer, intProc)
    !ERROR: Procedure pointer 'realprocpointer1' associated with incompatible procedure designator 'intproc'
    realProcPointer1 => intProc
    !ERROR: Procedure pointer 'realprocpointer1' associated with incompatible procedure designator 'intproc'
    lvar = associated(realProcPointer1, intProc)
    !ERROR: Procedure pointer 'subprocpointer' with explicit interface may not be associated with procedure designator 'externalproc' with implicit interface
    subProcPointer => externalProc
    !ERROR: Procedure pointer 'subprocpointer' with explicit interface may not be associated with procedure designator 'externalproc' with implicit interface
    lvar = associated(subProcPointer, externalProc)
    !ERROR: Subroutine pointer 'subprocpointer' may not be associated with function designator 'intproc'
    subProcPointer => intProc
    !ERROR: Subroutine pointer 'subprocpointer' may not be associated with function designator 'intproc'
    lvar = associated(subProcPointer, intProc)
    !ERROR: Function pointer 'intprocpointer1' may not be associated with subroutine designator 'subproc'
    intProcPointer1 => subProc
    !ERROR: Function pointer 'intprocpointer1' may not be associated with subroutine designator 'subproc'
    lvar = associated(intProcPointer1, subProc)
    !ERROR: Procedure pointer 'implicitprocpointer' with implicit interface may not be associated with procedure designator 'subr' with explicit interface
    implicitProcPointer => subr
    !ERROR: Procedure pointer 'implicitprocpointer' with implicit interface may not be associated with procedure designator 'subr' with explicit interface
    lvar = associated(implicitProcPointer, subr)
  end subroutine test
end subroutine assoc
