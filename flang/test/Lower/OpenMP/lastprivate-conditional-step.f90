! Test lowering of `lastprivate(conditional:)` on loops with a non-unit and a
! negative step.  The reduction struct records the CANONICAL (normalised)
! iteration index -- (iv - lb) / step -- not the loop-variable value, so that
! the combiner's "sequentially last" selection is correct regardless of step.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

! Non-unit positive step.
subroutine test_conditional_lp_step2(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i

  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n, 2
    if (mod(i, 3) == 0) x = i * 10
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_step2
! CHECK:         omp.loop_nest
! CHECK-SAME:      step (%c2{{[^)]*}})
! Canonical index = (iv - lb) / step, computed in i64 (operands widened first),
! stored into the $x index field.
! CHECK:           %[[STEP2:.*]] = fir.convert %c2{{[^ ]*}} : (i32) -> i64
! CHECK:           arith.subi %{{.*}}, %{{.*}} : i64
! CHECK:           %[[IDX2:.*]] = arith.divsi %{{.*}}, %[[STEP2]] : i64
! CHECK:           fir.if
! CHECK:             fir.coordinate_of %{{.*}}, $x
! CHECK:             fir.store %{{.*}} to %{{.*}} : !fir.ref<i64>

! Negative step.
subroutine test_conditional_lp_negstep(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i

  !$omp parallel do lastprivate(conditional: x)
  do i = n, 1, -1
    if (mod(i, 3) == 0) x = i * 10
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_negstep
! CHECK:         omp.loop_nest
! CHECK-SAME:      step (%c-1{{[^)]*}})
! Canonical index normalises by the negative step, so the last EXECUTED
! iteration gets the largest index.  Computed in i64 (operands widened first).
! CHECK:           %[[STEPN:.*]] = fir.convert %c-1{{[^ ]*}} : (i32) -> i64
! CHECK:           arith.subi %{{.*}}, %{{.*}} : i64
! CHECK:           arith.divsi %{{.*}}, %[[STEPN]] : i64
! CHECK:           fir.if
! CHECK:             fir.coordinate_of %{{.*}}, $x
! CHECK:             fir.store %{{.*}} to %{{.*}} : !fir.ref<i64>
