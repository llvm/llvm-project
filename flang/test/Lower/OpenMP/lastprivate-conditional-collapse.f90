! Test lowering of `lastprivate(conditional:)` on a collapsed loop nest.  The
! "sequentially last" iteration is over the COLLAPSED (flattened) iteration
! space, so the canonical index stored in the reduction struct must be the
! flattened index: outer_canonical * inner_extent + inner_canonical.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_collapse(n, m, x)
  implicit none
  integer, intent(in) :: n, m
  integer, intent(inout) :: x
  integer :: i, j

  !$omp parallel do collapse(2) lastprivate(conditional: x)
  do i = 1, n
    do j = 1, m
      if (mod(i + j, 2) == 1) x = i * 10 + j
    end do
  end do
  !$omp end parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_collapse
! CHECK:         omp.wsloop
! CHECK-SAME:      reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK:           omp.loop_nest {{.*}} collapse(2)
! Flattened canonical index: normalise each IV ((iv - lb) / step), then combine
! outer * inner_extent + inner, and store it into the $x index field.
! CHECK:             %[[OUTER:.*]] = arith.divsi
! CHECK:             %[[INNER:.*]] = arith.divsi
! CHECK:             %[[FLAT:.*]] = arith.muli %{{.*}}, %{{.*}} : i64
! CHECK:             %[[IDX:.*]] = arith.addi %[[FLAT]], %{{.*}} : i64
! CHECK:             fir.if
! CHECK:               fir.coordinate_of %{{.*}}, $x
! CHECK:               fir.store %[[IDX]] to %{{.*}} : !fir.ref<i64>
! CHECK:             }

! Guarded copy-back in an omp.single sibling.
! CHECK:           omp.single {
! CHECK:             arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:             omp.terminator
! CHECK:           }
