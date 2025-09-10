! Tests that when a loop bound is used in the body, that the mapped version of
! the loop bound (rather than the host-eval one) is the one used inside the loop.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=device %s -o - \
! RUN:   | FileCheck %s

subroutine foo(a, n)
  implicit none
  integer :: i, n
  real, dimension(n) :: a

  do concurrent (i=1:n)
    a(i) = n
  end do
end subroutine 

! CHECK: omp.target host_eval(%{{.*}} -> %{{.*}}, %{{.*}} -> %[[N_HOST_EVAL:.*]], %{{.*}} -> %{{.*}} : {{.*}}) map_entries({{.*}}) {
! CHECK:   %[[N_MAPPED:.*]]:2 = hlfir.declare %arg{{.*}} {uniq_name = "_QFfooEn"}
! CHECK:   omp.teams {
! CHECK:     omp.parallel {
! CHECK:       omp.distribute {
! CHECK:         omp.wsloop {
! CHECK:           omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[N_HOST_EVAL]]) inclusive step (%{{.*}}) {
! CHECK:             %[[N_VAL:.*]] = fir.load %[[N_MAPPED]]#0 : !fir.ref<i32>
! CHECK:             %[[N_VAL_CVT:.*]] = fir.convert %[[N_VAL]] : (i32) -> f32
! CHECK:             hlfir.assign %[[N_VAL_CVT]] to {{.*}}
! CHECK-NEXT:        omp.yield
! CHECK:           }
! CHECK:         }
! CHECK:       }
! CHECK:     }
! CHECK:   }
! CHECK: }
