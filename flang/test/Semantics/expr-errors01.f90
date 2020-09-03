! RUN: %S/test_errors.sh %s %t %f18
! C1003 - can't parenthesize function call returning procedure pointer
module m1
  type :: dt
    procedure(frpp), pointer, nopass :: pp
  end type dt
 contains
  subroutine boring
  end subroutine boring
  function frpp
    procedure(boring), pointer :: frpp
    frpp => boring
  end function frpp
  subroutine tests
    procedure(boring), pointer :: mypp
    type(dt) :: dtinst
    mypp => boring ! legal
    mypp => (boring) ! legal, not a function reference
    !ERROR: A function reference that returns a procedure pointer may not be parenthesized
    mypp => (frpp()) ! C1003
    mypp => frpp() ! legal, not parenthesized
    dtinst%pp => frpp
    mypp => dtinst%pp() ! legal
    !ERROR: A function reference that returns a procedure pointer may not be parenthesized
    mypp => (dtinst%pp())
  end subroutine tests
end module m1
