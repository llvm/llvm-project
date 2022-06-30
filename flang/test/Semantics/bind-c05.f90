! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for C1552
program main
 contains
  subroutine internal1() bind(c) ! ok
  end subroutine
  !ERROR: An internal procedure may not have a BIND(C,NAME=) binding label
  subroutine internal2() bind(c,name="internal2")
  end subroutine
  !ERROR: An internal procedure may not have a BIND(C,NAME=) binding label
  subroutine internal3() bind(c,name="")
  end subroutine
end
