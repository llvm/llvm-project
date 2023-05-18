! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for multiple symbols being defined with with same BIND(C) name

module m1
  integer, bind(c, name="x1") :: x1
  !ERROR: Two entities have the same global name 'x1'
  integer, bind(c, name=" x1 ") :: x2
 contains
  subroutine x3() bind(c, name="x3")
  end subroutine
end module

!ERROR: Two entities have the same global name 'x3'
subroutine x4() bind(c, name=" x3 ")
end subroutine

! Ensure no error in this situation
module m2
 interface
  subroutine x5() bind(c, name=" x5 ")
  end subroutine
 end interface
end module
subroutine x5() bind(c, name=" x5 ")
end subroutine

! Ensure no error in this situation
subroutine foo() bind(c, name="x6")
end subroutine
subroutine foo() bind(c, name="x7")
end subroutine
