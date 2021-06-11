! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Check for multiple symbols being defined with with same BIND(C) name

module m1
  integer, bind(c, name="x1") :: x1
  !ERROR: Two symbols have the same BIND(C) name 'x1'
  integer, bind(c, name=" x1 ") :: x2
 contains
  !ERROR: Two symbols have the same BIND(C) name 'x3'
  subroutine x3() bind(c, name="x3")
  end subroutine
end module

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
