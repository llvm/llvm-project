! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
! Tests of selectors whose defining expressions are pointer-valued functions;
! they must be valid targets, but not pointers.
! (F'2018 11.1.3.3 p1) "The associating entity does not have the ALLOCATABLE or
! POINTER attributes; it has the TARGET attribute if and only if the selector
! is a variable and has either the TARGET or POINTER attribute."
module m1
  type t
   contains
    procedure, nopass :: iptr
  end type
 contains
  function iptr(n)
    integer, intent(in), target :: n
    integer, pointer :: iptr
    !WARNING: Pointer target is not a definable variable
    !BECAUSE: 'n' is an INTENT(IN) dummy argument
    iptr => n
  end function
  subroutine test
    type(t) tv
    integer, target :: itarget
    integer, pointer :: ip
    associate (sel => iptr(itarget))
      ip => sel
      !ERROR: POINTER= argument of ASSOCIATED() must be a pointer
      if (.not. associated(sel)) stop
    end associate
    associate (sel => tv%iptr(itarget))
      ip => sel
      !ERROR: POINTER= argument of ASSOCIATED() must be a pointer
      if (.not. associated(sel)) stop
    end associate
    associate (sel => (iptr(itarget)))
      !ERROR: In assignment to object pointer 'ip', the target 'sel' is not an object with POINTER or TARGET attributes
      ip => sel
      !ERROR: POINTER= argument of ASSOCIATED() must be a pointer
      if (.not. associated(sel)) stop
    end associate
    associate (sel => 0 + iptr(itarget))
      !ERROR: In assignment to object pointer 'ip', the target 'sel' is not an object with POINTER or TARGET attributes
      ip => sel
      !ERROR: POINTER= argument of ASSOCIATED() must be a pointer
      if (.not. associated(sel)) stop
    end associate
  end subroutine
end module
