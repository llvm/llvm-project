! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine simd_ivs
  implicit none
  integer :: ido1 = 1
  integer :: ido2 = 2
  integer :: ido3 = 3
  integer :: ido4 = 4

  !$omp parallel
  !$omp simd collapse(3)
  do ido1 = 1, 10
    do ido2 = 1, 10
      do ido3 = 1, 10
        do ido4 = 1, 10
        end do
      end do
    end do
  end do
  !$omp end simd
  !$omp end parallel
end subroutine

! CHECK: func.func @_QPsimd_ivs() {
! CHECK:   %[[IDO1_HOST_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Eido1"}
! CHECK:   %[[IDO2_HOST_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Eido2"}
! CHECK:   %[[IDO3_HOST_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Eido3"}

! CHECK:   omp.parallel {
! CHECK:     omp.simd private(
! CHECK-SAME:  @{{.*}}do1_private{{.*}} %[[IDO1_HOST_DECL]]#0 -> %[[IDO1_PRIV_ARG:[^[:space:]]*]],
! CHECK-SAME:  @{{.*}}do2_private{{.*}} %[[IDO2_HOST_DECL]]#0 -> %[[IDO2_PRIV_ARG:[^[:space:]]*]],
! CHECK-SAME:  @{{.*}}do3_private{{.*}} %[[IDO3_HOST_DECL]]#0 -> %[[IDO3_PRIV_ARG:[^[:space:]]*]]
! CHECK-SAME:  : {{.*}}) {

! CHECK:       omp.loop_nest (%[[IV1:.*]], %[[IV2:.*]], %[[IV3:.*]]) : {{.*}} {
! CHECK:         %[[IDO1_PRIV_DECL:.*]]:2 = hlfir.declare %[[IDO1_PRIV_ARG]] {uniq_name = "{{.*}}Eido1"}
! CHECK:         %[[IDO2_PRIV_DECL:.*]]:2 = hlfir.declare %[[IDO2_PRIV_ARG]] {uniq_name = "{{.*}}Eido2"}
! CHECK:         %[[IDO3_PRIV_DECL:.*]]:2 = hlfir.declare %[[IDO3_PRIV_ARG]] {uniq_name = "{{.*}}Eido3"}

! CHECK:         fir.if %33 {
! CHECK:           fir.store %{{.*}} to %[[IDO1_PRIV_DECL]]#1
! CHECK:           fir.store %{{.*}} to %[[IDO2_PRIV_DECL]]#1
! CHECK:           fir.store %{{.*}} to %[[IDO3_PRIV_DECL]]#1
! CHECK:           %[[IDO1_VAL:.*]] = fir.load %[[IDO1_PRIV_DECL]]#0
! CHECK:           hlfir.assign %[[IDO1_VAL]] to %[[IDO1_HOST_DECL]]#0
! CHECK:           %[[IDO2_VAL:.*]] = fir.load %[[IDO2_PRIV_DECL]]#0
! CHECK:           hlfir.assign %[[IDO2_VAL]] to %[[IDO2_HOST_DECL]]#0
! CHECK:           %[[IDO3_VAL:.*]] = fir.load %[[IDO3_PRIV_DECL]]#0
! CHECK:           hlfir.assign %[[IDO3_VAL]] to %[[IDO3_HOST_DECL]]#0
! CHECK:         }
! CHECK-NEXT:    omp.yield
! CHECK:       }

! CHECK:     }

! CHECK:     omp.terminator
! CHECK:   }

! CHECK: }
