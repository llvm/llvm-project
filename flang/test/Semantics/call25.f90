! RUN: not %flang -fsyntax-only 2>&1 %s | FileCheck %s
module m
 contains
  subroutine subr1(f)
    character(5) f
    print *, f('abcde')
  end subroutine
  subroutine subr2(f)
    character(*) f
    print *, f('abcde')
  end subroutine
  character(5) function explicitLength(x)
    character(5), intent(in) :: x
    explicitLength = x
  end function
  real function notChar(x)
    character(*), intent(in) :: x
    notChar = 0
  end function
end module

character(*) function assumedLength(x)
  character(*), intent(in) :: x
  assumedLength = x
end function

subroutine subr3(f)
  character(5) f
  print *, f('abcde')
end subroutine

program main
  use m
  external assumedlength
  character(5) :: assumedlength
  call subr1(explicitLength)
  call subr1(assumedLength)
  !CHECK: error: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
  call subr1(notChar)
  call subr2(explicitLength)
  call subr2(assumedLength)
  !CHECK: error: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
  call subr2(notChar)
  call subr3(explicitLength)
  call subr3(assumedLength)
  !CHECK: warning: If the procedure's interface were explicit, this reference would be in error
  !CHECK: because: Actual argument function associated with procedure dummy argument 'f=' has incompatible result type
  call subr3(notChar)
end program
