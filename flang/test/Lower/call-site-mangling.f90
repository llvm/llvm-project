! RUN: bbc %s -o "-" -emit-fir | FileCheck %s

subroutine sub()
  real :: x
  ! CHECK-LABEL: fir.call @_QPasubroutine()
  call AsUbRoUtInE();
  ! CHECK-LABEL: fir.call @_QPfoo()
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
  ! CHECK-LABEL: fir.call @_QMtestmodPsub()
  call Sub();
  ! CHECK-LABEL: fir.call @_QMtestmodPfoo()
  x = foo()
end subroutine

subroutine sub2()
  use testMod, localfoo => foo, localsub => sub
  real :: x
  ! CHECK-LABEL: fir.call @_QMtestmodPsub()
  call localsub();
  ! CHECK-LABEL: fir.call @_QMtestmodPfoo()
  x = localfoo()
end subroutine



subroutine sub3()
  real :: x
  ! CHECK-LABEL: fir.call @_QFsub3Psub()
  call sub();
  ! CHECK-LABEL: fir.call @_QFsub3Pfoo()
  x = foo()
contains
  subroutine sub()
  end subroutine

  function foo()
  end function
end subroutine
