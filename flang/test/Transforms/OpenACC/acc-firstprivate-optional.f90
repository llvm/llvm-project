! Test lowering of firstprivate on optional dummy arguments.
! Optional variables use a separate recipe with _optional suffix and null-check
! in init/copy regions.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

subroutine test_optional_firstprivate(x)
  integer, optional :: x

  !$acc parallel firstprivate(x)
  !$acc end parallel
end subroutine

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_optional_ref_i32 : !fir.ref<i32> init {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:         fir.if {{.*}} -> (!fir.ref<i32>) {
! CHECK:           %[[ALLOC:.*]] = fir.alloca i32
! CHECK:           fir.result %[[ALLOC]] : !fir.ref<i32>
! CHECK:         } else {
! CHECK:           fir.absent !fir.ref<i32>
! CHECK:           fir.result
! CHECK:         }
! CHECK:         acc.yield {{.*}} : !fir.ref<i32>
! CHECK:       } copy {
! CHECK:       ^bb0(%[[SRC:.*]]: !fir.ref<i32>, %[[DST:.*]]: !fir.ref<i32>):
! CHECK:         fir.if {{.*}} {
! CHECK:           {{.*}} = fir.load %[[SRC]] : !fir.ref<i32>
! CHECK:           hlfir.assign {{.*}} to %[[DST]] {{.*}}
! CHECK:         }
! CHECK:         acc.terminator
! CHECK:       }
