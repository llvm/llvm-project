! Test lowering of `lastprivate(conditional:)` on an omp sections construct
! with the nowait clause.  The lowering must emit an explicit barrier
! before the copy-back to ensure the reduction is fully finalized.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_sections_nowait(x)
  implicit none
  integer, intent(inout) :: x

  !$omp parallel
  !$omp sections lastprivate(conditional: x) nowait
    !$omp section
    x = 10
    !$omp section
    x = 20
  !$omp end sections nowait
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_sections_nowait
! CHECK:         %[[STRUCT:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>

! CHECK:         omp.parallel {
! CHECK:           omp.sections nowait
! CHECK-SAME:        reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK-SAME:          %[[STRUCT]]

! CHECK:           omp.barrier

! CHECK:           omp.single {
! CHECK:             fir.coordinate_of %[[STRUCT]], x
! CHECK:             fir.load
! CHECK:             fir.coordinate_of %[[STRUCT]], $x
! CHECK:             fir.load
! CHECK:             arith.cmpi sge
! CHECK:             fir.if
! CHECK:               fir.store
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
