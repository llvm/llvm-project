! Test that -ffpe-trap= only affects the main program unit: a compilation unit
! without a main program must not generate a call to _FortranAEnableFPETraps.

! RUN: %flang_fc1 -emit-fir -ffpe-trap=invalid,zero,overflow %s -o - | FileCheck %s

subroutine sub(x)
  real :: x
  x = x + 1.0
end subroutine

module m
contains
  function f(y) result(z)
    real :: y, z
    z = y * 2.0
  end function
end module

! CHECK-NOT: fir.call @_FortranAEnableFPETraps
! CHECK-NOT: @_QQmain
