! Test lowering of `lastprivate(conditional:)` on a worksharing do loop
! with the nowait clause.  The lowering must emit an explicit barrier
! before the copy-back to ensure the reduction is fully finalized.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_conditional_lp_nowait(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: k

  !$omp parallel
  !$omp do lastprivate(conditional: x) nowait
  do k = 1, n
    x = k
  end do
  !$omp end do nowait
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_conditional_lp_nowait
! CHECK:         %[[STRUCT:.*]] = fir.alloca !fir.type<_lp_cond_t.{{l[0-9]+\.[0-9]+}}{x:i32,$x:i64}>

! CHECK:         omp.parallel {
! CHECK:           omp.wsloop nowait
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
