!RUN: split-file %s %t

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/no_bind_clause.f90 -o - \
!RUN: | FileCheck %s

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 %t/bind_clause_teams.f90 -o - \
!RUN: | FileCheck %s

!--- no_bind_clause.f90
subroutine target_teams_loop
    implicit none
    integer :: x, i

    !$omp target teams loop
    do i = 0, 10
      x = x + i
    end do
end subroutine target_teams_loop

!--- bind_clause_teams.f90
subroutine target_teams_loop
    implicit none
    integer :: x, i

    !$omp target teams loop bind(teams)
    do i = 0, 10
      x = x + i
    end do
end subroutine target_teams_loop

!CHECK-LABEL: func.func @_QPtarget_teams_loop
!CHECK:         omp.target map_entries(
!CHECK-SAME:      %{{.*}} -> %[[I_ARG:[^[:space:]]+]],
!CHECK-SAME:      %{{.*}} -> %[[X_ARG:[^[:space:]]+]] : {{.*}}) {

!CHECK:           %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ARG]]
!CHECK:           %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]]

!CHECK:           omp.teams {

!CHECK:             %[[LB:.*]] = arith.constant 0 : i32
!CHECK:             %[[UB:.*]] = arith.constant 10 : i32
!CHECK:             %[[STEP:.*]] = arith.constant 1 : i32

!CHECK:             omp.parallel private(@{{.*}} %[[I_DECL]]#0 
!CHECK-SAME:          -> %[[I_PRIV_ARG:[^[:space:]]+]] : !fir.ref<i32>) {
!CHECK:               omp.distribute {
!CHECK:                 omp.wsloop {

!CHECK:                   omp.loop_nest (%{{.*}}) : i32 = 
!CHECK-SAME:                (%[[LB]]) to (%[[UB]]) inclusive step (%[[STEP]]) {
!CHECK:                     %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_PRIV_ARG]]
!CHECK:                     fir.store %{{.*}} to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
!CHECK:                   }
!CHECK:                 }
!CHECK:               }
!CHECK:             }
!CHECK:           }
!CHECK:         }
