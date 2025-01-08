! Test -fsave-main-program switch.
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-DEFAULT %s
! RUN: %flang_fc1 -fsave-main-program -emit-hlfir -o - %s | FileCheck --check-prefix=CHECK-SAVE %s
program test
integer :: i
call foo(i)
end

!CHECK-DEFAULT-NOT: fir.global internal @_QFEi
!CHECK-SAVE: fir.global internal @_QFEi
