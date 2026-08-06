!RUN: %python %S/test_errors.py %s %flang_fc1
!Test that a self-referencing character length specification in a function
!prefix does not cause infinite recursion (crash) in the compiler.

!ERROR: 'f1' is already declared in this scoping unit
character(f1) function f1()
  implicit integer(f)
  f = 2
end function

!ERROR: 'f2' is not a callable procedure
!ERROR: Use of 'f2' as a procedure conflicts with its declaration
character(f2(1)) function f2()
  implicit integer(f)
  f = 2
end function

!ERROR: 'f3' is already declared in this scoping unit
integer(f3) function f3()
  f3 = 2
end function

!ERROR: 'f4' is already declared in this scoping unit
character*(f4) function bb() result(f4)
f4 = "a"
end
