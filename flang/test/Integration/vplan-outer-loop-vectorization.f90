! When a `!dir$ vector always` loop encloses an inner loop with a loop-carried
! dependence, only the outer loop can be vectorized. flang normally handles this
! with the VectorAlwaysUnroll pass, which fully unrolls the (constant-trip)
! inner loop so the ordinary (inner-loop) vectorizer can then vectorize the
! annotated outer loop.
!
! This test disables that workaround (-disable-vector-always-unroll) to check
! whether LLVM's VPlan-native outer-loop vectorization can vectorize the outer
! loop directly. -fno-unroll-loops keeps LLVM's own unroller from unrolling the
! inner loop on its own (which would otherwise expose the outer loop to the
! regular vectorizer and mask the need for the workaround); with it, only a
! forced `llvm.loop.unroll.full` from the workaround unrolls the inner loop.
!
! VPlan-native outer-loop vectorization does not yet handle this pattern, so the
! expected "vectorized loop" remark is not produced and the test is marked
! XFAIL. When that path learns to vectorize this loop, the remark appears and
! the test starts passing (XPASS). At that point:
!   * drop the XFAIL below, and
!   * the VectorAlwaysUnroll workaround and its scheduling in
!     flang/lib/Optimizer/Passes/Pipelines.cpp can be removed.

! REQUIRES: x86-registered-target
! XFAIL: *

! RUN: %flang_fc1 -emit-llvm -O2 -triple x86_64-unknown-linux-gnu \
! RUN:   -mllvm -enable-vplan-native-path -mmlir -disable-vector-always-unroll \
! RUN:   -fno-unroll-loops -Rpass=loop-vectorize -o /dev/null %s 2>&1 \
! RUN:   | FileCheck %s

subroutine outer_vec(a, b)
  real :: a(8, 8), b(8, 8)
  integer :: i, j
  !dir$ vector always
  do i = 1, 8
     do j = 2, 8
        a(j, i) = a(j - 1, i) + b(j, i)
     end do
  end do
end subroutine outer_vec

! CHECK: remark: {{.*}}vectorized loop
