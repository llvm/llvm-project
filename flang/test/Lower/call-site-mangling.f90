! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub()
  real :: x
  ! CHECK: fir.call @_QPasubroutine()
  call AsUbRoUtInE();
  ! CHECK: fir.call @_QPfoo()
  x = foo()
end subroutine

module testMod
contains
  subroutine sub()
  end subroutine

  function foo()
  end function
end module

subroutine sub1()
  use testMod
  real :: x
  ! CHECK: fir.call @_QMtestmodPsub()
  call Sub();
  ! CHECK: fir.call @_QMtestmodPfoo()
  x = foo()
end subroutine

subroutine sub2()
  use testMod, localfoo => foo, localsub => sub
  real :: x
  ! CHECK: fir.call @_QMtestmodPsub()
  call localsub();
  ! CHECK: fir.call @_QMtestmodPfoo()
  x = localfoo()
end subroutine



subroutine sub3()
  real :: x
  ! CHECK: fir.call @_QFsub3Psub()
  call sub();
  ! CHECK: fir.call @_QFsub3Pfoo()
  x = foo()
contains
  subroutine sub()
  end subroutine

  function foo()
  end function
end subroutine

function foo1()
  real :: bar1
  ! CHECK: fir.call @_QPbar1()
  foo1 = bar1()
end function

function foo2()
  ! CHECK: fir.call @_QPbar2()
  foo2 = bar2()
end function

function foo3()
  interface
  real function bar3()
  end function
  end interface
  ! CHECK: fir.call @_QPbar3()
  foo3 = bar3()
end function

function foo4()
  external :: bar4
  ! CHECK: fir.call @_QPbar4()
  foo4 = bar4()
end function

