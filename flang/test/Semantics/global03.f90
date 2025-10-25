! RUN: %python %S/test_errors.py %s %flang_fc1 -Werror
! Catch discrepancies between implicit result types and a global definition,
! allowing for derived types and type equivalence.

module m
  type t1
    integer n
  end type
  type t2
    real a
  end type
  type t3
    sequence
    integer n
  end type
end

function xfunc1()
  use m
  type(t1) xfunc1
  xfunc1%n = 123
end

function yfunc1()
  use m
  type(t1) yfunc1
  yfunc1%n = 123
end

function zfunc1()
  type t3
    sequence
    integer n
  end type
  type(t3) zfunc1
  zfunc1%n = 123
end

program main
  use m
  implicit type(t1) (x)
  implicit type(t2) (y)
  implicit type(t3) (z)
  print *, xfunc1() ! ok
  print *, xfunc2() ! ok
!ERROR: Implicit declaration of function 'yfunc1' has a different result type than in previous declaration
  print *, yfunc1()
  print *, yfunc2()
  print *, zfunc1() ! ok
  print *, zfunc2() ! ok
end

function xfunc2()
  use m
  type(t1) xfunc2
  xfunc2%n = 123
end

function yfunc2()
  use m
!ERROR: Function 'yfunc2' has a result type that differs from the implicit type it obtained in a previous reference
  type(t1) yfunc2
  yfunc2%n = 123
end

function zfunc2()
  type t3
    sequence
    integer n
  end type
  type(t3) zfunc2
  zfunc2%n = 123
end
