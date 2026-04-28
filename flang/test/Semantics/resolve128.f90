! RUN: not %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Test that a self-referencing character length specification in a function
! prefix does not cause infinite recursion (crash) in the compiler.

! CHECK: error: Function cannot have both an explicit type prefix and a RESULT suffix
character(f1) function f1()
  implicit integer(f)
  f = 2
end function

! CHECK: error: Use of 'f2' as a procedure conflicts with its declaration
character(f2(1)) function f2()
  implicit integer(f)
  f = 2
end function

! CHECK: error: Must be a constant value
integer(f3) function f3()
  f3 = 2
end function
