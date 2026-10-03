! Test that -fstack-arrays is not applied to acc routines. They are compiled for
! the device as well, where the stack is far smaller than on the host, so
! lowering records the same policy as for a device procedure. The policy is
! recorded whether or not -fstack-arrays was requested, so that the opt-out is
! explicit in the IR.

! RUN: %flang_fc1 -emit-fir -fopenacc %s -o - | FileCheck %s --check-prefixes=CHECK,DEFAULT
! RUN: %flang_fc1 -emit-fir -fopenacc -fstack-arrays %s -o - | FileCheck %s --check-prefixes=CHECK,STACK

subroutine routine_seq(a, n)
  !$acc routine seq
  integer :: a(*)
  integer :: n
  integer :: auto(n)
  do i = 1, n
    auto(i) = i
  end do
  a(1) = sum(auto(1:n))
end subroutine

subroutine host_sub(a, n)
  integer :: a(*)
  integer :: n
  integer :: auto(n)
  do i = 1, n
    auto(i) = i
  end do
  a(1) = sum(auto(1:n))
end subroutine

! The module policy follows -fstack-arrays.
! DEFAULT: module attributes {{.*}}fir.allocation_policy = #fir.allocation_policy<stack_arrays = false
! STACK: module attributes {{.*}}fir.allocation_policy = #fir.allocation_policy<stack_arrays = true

! The acc routine opts out of it in both cases.
! CHECK: func.func @_QProutine_seq({{.*}}) attributes {acc.routine_info = #acc.routine_info<[@acc_routine_0]>, fir.allocation_policy = #fir.allocation_policy<stack_arrays = false

! The host procedure keeps the module policy.
! CHECK: func.func @_QPhost_sub({{.*}}) {
