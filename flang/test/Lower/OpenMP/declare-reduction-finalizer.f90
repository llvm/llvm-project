! Test declare reduction with derived types that have FINAL subroutines
! and non-trivial user-defined initializers, to verify that initialization
! and finalization are generated correctly.
!
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! ---------------------------------------------------------------------
! Test 1: Simple derived type with finalizer and structure constructor init
! ---------------------------------------------------------------------
module m1
  implicit none

  type :: t
    integer :: x = -999
  contains
    final :: cleanup
  end type t

contains

  subroutine cleanup(this)
    type(t), intent(inout) :: this
    this%x = 0
  end subroutine cleanup

end module m1

! CHECK-LABEL: omp.declare_reduction @plus_t{{.*}} : !fir.ref<{{.*}}>
!
! -- alloc region
! CHECK:        alloc {
! CHECK:          %[[ALLOCA:.*]] = fir.alloca
! CHECK:          omp.yield(%[[ALLOCA]] :
!
! -- init region: must store 100 (from initializer clause), not -999 (default)
! CHECK:        } init {
! CHECK:        ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<{{.*}}>, %[[INIT_ARG1:.*]]: !fir.ref<{{.*}}>):
! CHECK:          %{{.*}}:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"}
! CHECK:          %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[INIT_ARG1]] {uniq_name = "omp_priv"}
! CHECK:          %[[INIT_ADDR:.*]] = fir.address_of(@_QQro._QMm1Tt.0)
! CHECK:          %[[INIT_DECL:.*]]:2 = hlfir.declare %[[INIT_ADDR]]
! CHECK:          %[[INIT_VAL:.*]] = fir.load %[[INIT_DECL]]#0
! CHECK:          fir.store %[[INIT_VAL]] to %[[INIT_ARG1]]
! CHECK:          omp.yield(%[[INIT_ARG1]] :
!
! -- combiner region
! CHECK:        } combiner {
! CHECK:        ^bb0(%[[LHS:.*]]: !fir.ref<{{.*}}>, %[[RHS:.*]]: !fir.ref<{{.*}}>):
! CHECK:          %{{.*}}:2 = hlfir.declare %[[RHS]] {uniq_name = "omp_in"}
! CHECK:          %{{.*}}:2 = hlfir.declare %[[LHS]] {uniq_name = "omp_out"}
! CHECK:          hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK:          omp.yield(%[[LHS]] :
! -- cleanup region: calls runtime destroy (which dispatches to the finalizer)
! CHECK:        } cleanup {
! CHECK:        ^bb0(%[[CLEANUP_ARG:.*]]: !fir.ref<{{.*}}>):
! CHECK:          %[[BOX:.*]] = fir.embox %[[CLEANUP_ARG]]
! CHECK:          %[[CONV:.*]] = fir.convert %[[BOX]]
! CHECK:          fir.call @_FortranADestroy(%[[CONV]])
! CHECK:          omp.yield
! CHECK:        }
!
! TODO: Test declare reduction without an initializer clause to verify
! the default constructor value (-999) is used. This requires support
! for declare reduction without an initializer clause.

! Verify the init value constant is 100 (from T(100)), not -999 (default)
! CHECK: fir.global internal @_QQro._QMm1Tt.0 constant
! CHECK:   %[[C100:.*]] = arith.constant 100 : i32
! CHECK:   fir.insert_value %{{.*}}, %[[C100]], ["x",

program test1
  use m1
  implicit none

  type(t) :: a

  !$omp declare reduction(plus_t:t: omp_out%x = omp_out%x + omp_in%x) &
  !$omp&  initializer(omp_priv = t(100))

  a = t(200)

  !$omp parallel reduction(plus_t:a)
  a%x = a%x + 1
  !$omp end parallel

end program test1
