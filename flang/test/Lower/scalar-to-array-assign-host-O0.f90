! Regression test for the follow-up to PR llvm/llvm-project#197092.
!
! At -O0 on the host (no OpenMP target-device compilation), a scalar-to-array
! broadcast assignment must lower to a Fortran runtime call
! (_FortranAAssign), not to an inline assignment loop. Lowering it inline
! at -O0 caused -g line breakpoints to hit once per array element instead
! of once.

! RUN: %flang_fc1 -emit-fir -O0 %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPhost_scalar_broadcast
subroutine host_scalar_broadcast(arr)
  integer :: arr(4)
  ! CHECK: fir.call @_FortranAAssign
  ! CHECK-NOT: fir.do_loop
  arr = 11
end subroutine
