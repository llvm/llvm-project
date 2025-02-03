! RUN: %flang_fc1 -fopenmp -mmlir --openmp-enable-delayed-privatization-staging=true -emit-hlfir %s -o - | FileCheck %s

subroutine first_and_lastprivate
  integer i
  integer :: var = 1

  !$omp parallel do firstprivate(var) lastprivate(var)
  do i=1,1
  end do
  !$omp end parallel do
end subroutine

! CHECK:  omp.private {type = firstprivate} @{{.*}}Evar_firstprivate_i32 : {{.*}} copy {
! CHECK: ^{{.*}}(%[[ORIG_REF:.*]]: {{.*}}, %[[PRIV_REF:.*]]: {{.*}}):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[ORIG_REF]]
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_REF]]
! CHECK:    omp.yield(%[[PRIV_REF]] : !fir.ref<i32>)
! CHECK:  }

! CHECK:  func.func @{{.*}}first_and_lastprivate()
! CHECK:    %[[ORIG_VAR_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Evar"}
! CHECK:    omp.parallel {
! CHECK:      omp.barrier
! CHECK:      omp.wsloop private(@{{.*}}var_firstprivate_i32 {{.*}}) {
! CHECK:        omp.loop_nest {{.*}} {
! CHECK:          %[[PRIV_VAR_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Evar"}
! CHECK:          fir.if %{{.*}} {
! CHECK:            %[[PRIV_VAR_VAL:.*]] = fir.load %[[PRIV_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:            hlfir.assign %[[PRIV_VAR_VAL]] to %[[ORIG_VAR_DECL]]#0
! CHECK:          }
! CHECK:          omp.yield
! CHECK:        }
! CHECK:      }
! CHECK:      omp.terminator
! CHECK:    }
