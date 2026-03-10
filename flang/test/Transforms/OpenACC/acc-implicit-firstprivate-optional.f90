! Test implicit firstprivate for optional dummy arguments.
! An optional scalar used in acc parallel without an explicit data clause gets
! implicit firstprivate. The recipe must use _optional suffix and emit fir.if
! null-check in init/copy regions.

! RUN: bbc -fopenacc -emit-hlfir %s -o - \
! RUN:   | fir-opt --pass-pipeline="builtin.module(acc-initialize-fir-analyses,acc-implicit-data)" \
! RUN:   -o - | FileCheck %s

subroutine test_optional_implicit_firstprivate(x)
  integer, optional :: x
  integer :: val

  ! No explicit firstprivate - x gets implicit firstprivate as a scalar in parallel
  ! x must be used in the region; assign to val so x becomes a live-in
  !$acc parallel
    val = 0
    if (present(x)) val = x
  !$acc end parallel
end subroutine

! CHECK-LABEL: acc.firstprivate.recipe @firstprivatization_optional_ref_i32
! CHECK:       ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:         fir.if {{.*}} -> (!fir.ref<i32>) {
! CHECK:           {{.*}} = fir.alloca i32
! CHECK:           fir.result {{.*}} : !fir.ref<i32>
! CHECK:         } else {
! CHECK:           {{.*}} = fir.zero_bits !fir.ref<i32>
! CHECK:           fir.result
! CHECK:         }
! CHECK:         acc.yield {{.*}} : !fir.ref<i32>
! CHECK:       } copy {
! CHECK:       ^bb0(%{{.*}}: !fir.ref<i32>, %{{.*}}: !fir.ref<i32>):
! CHECK:         fir.if {{.*}} {
! CHECK:           {{.*}} = fir.load {{.*}} : !fir.ref<i32>
! CHECK:           hlfir.assign {{.*}} to {{.*}} {{.*}}
! CHECK:         }
! CHECK:         acc.terminator
! CHECK:       }

! CHECK: acc.firstprivate varPtr(%{{.*}} : !fir.ref<i32>) recipe(@firstprivatization_optional_ref_i32) -> !fir.ref<i32> {implicit = true, name = "x"}
! CHECK: acc.parallel firstprivate(%{{.*}}, %{{.*}} : !fir.ref<i32>, !fir.ref<i32>)
