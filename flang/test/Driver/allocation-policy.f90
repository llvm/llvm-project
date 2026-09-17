! Test that lowering records the array allocation policy on the module.
! Policy-aware passes read it from there, and emitted FIR records the concrete
! policy values with which it was compiled.

! RUN: %flang_fc1 -emit-fir -o - %s | FileCheck %s --check-prefix=DEFAULT
! RUN: %flang_fc1 -emit-fir -mllvm -allocation-placement-small-array-size=2048 \
! RUN:   -mllvm -allocation-placement-stack-limit=8192 -o - %s \
! RUN:   | FileCheck %s --check-prefix=TUNED
! RUN: %flang_fc1 -emit-fir -fstack-arrays -o - %s \
! RUN:   | FileCheck %s --check-prefix=STACK

! The policy is recorded by the lowering bridge, so every tool that lowers
! Fortran gets it, not just the frontend driver.
! RUN: bbc -emit-fir %s -o - | FileCheck %s --check-prefix=DEFAULT
! RUN: bbc -emit-fir -allocation-placement-small-array-size=2048 \
! RUN:   -allocation-placement-stack-limit=8192 %s -o - \
! RUN:   | FileCheck %s --check-prefix=TUNED

! Default values are written explicitly so that a tool reading this FIR does
! not silently pick up different defaults.
! DEFAULT: fir.allocation_policy = #fir.allocation_policy<stack_arrays = false,
! DEFAULT-SAME: small_array_threshold = 1024,
! DEFAULT-SAME: total_stack_limit = 4194304>

! TUNED: fir.allocation_policy = #fir.allocation_policy<stack_arrays = false,
! TUNED-SAME: small_array_threshold = 2048, total_stack_limit = 8192>

! STACK: fir.allocation_policy = #fir.allocation_policy<stack_arrays = true,
! STACK-SAME: small_array_threshold = 1024,
! STACK-SAME: total_stack_limit = 4194304>

subroutine s(a)
  real :: a(10)
  a = 1.0
end subroutine s
