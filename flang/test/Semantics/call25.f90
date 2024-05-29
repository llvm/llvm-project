! RUN: not %flang -fsyntax-only -pedantic 2>&1 %s | FileCheck %s
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
  character(6) function badExplicitLength(x)
    character(5), intent(in) :: x
    badExplicitLength = x
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
  !CHECK: error: Actual argument function associated with procedure dummy argument 'f=' is not compatible: function results have distinct types: CHARACTER(KIND=1,LEN=5_8) vs CHARACTER(KIND=1,LEN=6_8)
  call subr1(badExplicitLength)
  call subr1(assumedLength)
  !CHECK: error: Actual argument function associated with procedure dummy argument 'f=' is not compatible: function results have distinct types: CHARACTER(KIND=1,LEN=5_8) vs REAL(4)
  call subr1(notChar)
  call subr2(explicitLength)
  call subr2(assumedLength)
  !CHECK: error: Actual argument function associated with procedure dummy argument 'f=' is not compatible: function results have distinct types: CHARACTER(KIND=1,LEN=*) vs REAL(4)
  call subr2(notChar)
  call subr3(explicitLength)
  !CHECK: warning: If the procedure's interface were explicit, this reference would be in error
  !CHECK: because: Actual argument function associated with procedure dummy argument 'f=' is not compatible: function results have distinct types: CHARACTER(KIND=1,LEN=5_8) vs CHARACTER(KIND=1,LEN=6_8)
  call subr3(badExplicitLength)
  call subr3(assumedLength)
  !CHECK: warning: If the procedure's interface were explicit, this reference would be in error
  !CHECK: because: Actual argument function associated with procedure dummy argument 'f=' is not compatible: function results have distinct types: CHARACTER(KIND=1,LEN=5_8) vs REAL(4)
  call subr3(notChar)
end program
