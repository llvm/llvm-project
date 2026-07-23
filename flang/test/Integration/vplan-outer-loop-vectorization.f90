! When a `!dir$ vector always` loop encloses another loop, flang relies on the
! VectorAlwaysUnroll pass to fully unroll the inner loop(s) so that the ordinary
! (inner-loop) vectorizer can vectorize the annotated loop. That workaround only
! applies when the inner loops have compile-time-constant trip counts.
!
! The loop below deliberately has a runtime inner trip count and a loop-carried
! dependence on the inner loop, so the VectorAlwaysUnroll workaround does not
! apply (even though it now runs regardless of `-enable-vplan-native-path`). The
! annotated outer loop can therefore only be vectorized by VPlan-native
! outer-loop vectorization. That path does not yet handle this pattern, so the
! forced vectorization request fails and LLVM emits the "unable to perform the
! requested transformation" warning checked below.
!
! This warning is a stable, loop-specific signal (only a forced `vector always`
! loop produces it), so the test asserts it directly rather than relying on the
! absence of an incidental remark via XFAIL. When VPlan-native outer-loop
! vectorization learns to vectorize the loop below, this warning disappears and
! the CHECK starts failing. At that point:
!   * update this test to check for the vectorization instead, and
!   * the VectorAlwaysUnroll workaround and its pipeline scheduling in
!     flang/lib/Optimizer/Passes/Pipelines.cpp can be removed.

! REQUIRES: x86-registered-target

! RUN: %flang_fc1 -emit-llvm -O2 -triple x86_64-unknown-linux-gnu \
! RUN:   -mllvm -enable-vplan-native-path -Rpass=loop-vectorize \
! RUN:   -o /dev/null %s 2>&1 | FileCheck %s

subroutine outer_vec(a, b, n)
  integer :: n, i, j
  real :: a(n, n), b(n, n)
  ! The inner loop over j carries a dependence (a(j,i) reads a(j-1,i)), so it
  ! is not a legal inner-loop vectorization candidate. Only the outer loop over
  ! i (independent columns) can be vectorized, and it is the one annotated.
  !dir$ vector always
  do i = 1, n
     do j = 2, n
        a(j, i) = a(j - 1, i) + b(j, i)
     end do
  end do
end subroutine outer_vec

! CHECK: loop not vectorized: the optimizer was unable to perform the requested transformation
