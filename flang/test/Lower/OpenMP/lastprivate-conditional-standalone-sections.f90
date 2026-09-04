! Test lowering of `lastprivate(conditional:)` on a STANDALONE `omp sections`
! nested inside a separate `omp parallel` (as opposed to the combined
! `parallel sections`), where the assignment inside each section is guarded by
! an `if`.  The canonical section-index store must be emitted INSIDE the fir.if,
! so a section that does not take its branch records no index (its slot keeps
! the -1 sentinel) and is not copied back.  (All other sections tests use
! unconditional assignments, where the index store is not guarded.)

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_sections_if(sel, x)
  implicit none
  integer, intent(in) :: sel
  integer, intent(inout) :: x

  !$omp parallel
  !$omp sections lastprivate(conditional: x)
  !$omp section
    if (sel > 0) x = 11
  !$omp section
    if (sel < 0) x = 22
  !$omp end sections
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_sections_if
! CHECK:         omp.parallel {
! CHECK:           omp.sections
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! Section 1: the value store and the $x index store are both inside the fir.if,
! so the section only records its index when its branch is taken.
! CHECK:           fir.if
! CHECK:             hlfir.assign
! CHECK:             fir.coordinate_of %{{.*}}, $x
! CHECK:             fir.store %{{.*}} to %{{.*}} : !fir.ref<i64>
! CHECK:           }
! Section 2 similarly guards its index store.
! CHECK:           fir.if
! CHECK:             hlfir.assign
! CHECK:             fir.coordinate_of %{{.*}}, $x
! CHECK:             fir.store %{{.*}} to %{{.*}} : !fir.ref<i64>
! CHECK:           }
! Guarded copy-back in an omp.single sibling.
! CHECK:           omp.single {
! CHECK:             arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:             omp.terminator
! CHECK:           }
