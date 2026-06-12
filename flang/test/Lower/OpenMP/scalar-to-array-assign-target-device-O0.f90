! Regression test for PR llvm/llvm-project#197092 and its follow-up.
!
! When compiling for an OpenMP target device at -O0, a scalar-to-array
! broadcast assignment inside a target region must still be inlined to
! avoid emitting a _FortranAAssign runtime call (which internally uses
! malloc/free) into GPU device code.

! RUN: %flang_fc1 -emit-fir -O0 -fopenmp -fopenmp-is-target-device %s -o - \
! RUN:   | FileCheck %s --implicit-check-not="fir.call @_FortranAAssign"

subroutine device_scalar_broadcast()
  integer :: arr(4)
  !$omp target map(tofrom: arr)
  ! CHECK: omp.target
  ! CHECK: fir.do_loop
  arr = 11
  !$omp end target
end subroutine
