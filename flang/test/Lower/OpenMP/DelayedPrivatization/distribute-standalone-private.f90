! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine standalone_distribute
    implicit none
    integer :: simple_var, i

    !$omp teams
    !$omp distribute private(simple_var)
    do i = 1, 10
      simple_var = simple_var + i
    end do
    !$omp end distribute
    !$omp end teams
end subroutine standalone_distribute

! CHECK: omp.private {type = private} @[[I_PRIVATIZER_SYM:.*]] : !fir.ref<i32>
! CHECK: omp.private {type = private} @[[VAR_PRIVATIZER_SYM:.*]] : !fir.ref<i32>


! CHECK-LABEL: func.func @_QPstandalone_distribute() {
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFstandalone_distributeEi"}
! CHECK:         %[[VAR_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFstandalone_distributeEsimple_var"}
! CHECK:         omp.teams {
! CHECK:           omp.distribute
! CHECK-SAME:        private(@[[VAR_PRIVATIZER_SYM]] %[[VAR_DECL]]#0 -> %[[VAR_ARG:.*]] : !fir.ref<i32>,
! CHECK-SAME:                @[[I_PRIVATIZER_SYM]] %[[I_DECL]]#0 -> %[[I_ARG:.*]] : !fir.ref<i32>) {
! CHECK:             omp.loop_nest {{.*}} {
! CHECK:               %[[VAR_PRIV_DECL:.*]]:2 = hlfir.declare %[[VAR_ARG]]
! CHECK:               %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_ARG]]

! CHECK:               fir.store %{{.*}} to %[[I_PRIV_DECL]]#1 : !fir.ref<i32>
! CHECK:               %{{.*}} = fir.load %[[VAR_PRIV_DECL]]#0 : !fir.ref<i32>
! CHECK:               %{{.*}} = fir.load %[[I_PRIV_DECL]]#0 : !fir.ref<i32>
! CHECK:               arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:               hlfir.assign %{{.*}} to %[[VAR_PRIV_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:         }
! CHECK:       }
