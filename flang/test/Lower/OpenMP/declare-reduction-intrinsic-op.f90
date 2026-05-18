! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

program test
  type t
     integer :: x
  end type t
  !$omp declare reduction(+:t: omp_out%x = omp_out%x + omp_in%x) initializer(omp_priv = t(0))
  type(t) :: a
  a = t(0)
  !$omp parallel reduction(+:a)
  a%x = a%x + 1
  !$omp end parallel
end program test

! CHECK: omp.declare_reduction @add_reduction_byref_rec__QFTt :
! CHECK:   %[[ALLOCA:.*]] = fir.alloca [[TY:.*]]
! CHECK:   omp.yield(%[[ALLOCA]] : !fir.ref<[[TY]]>)
! CHECK: } init {
! CHECK: ^bb0(%[[INIT_ARG0:.*]]: !fir.ref<[[TY]]>, %[[INIT_ARG1:.*]]: !fir.ref<[[TY]]>):
! CHECK:   %{{.*}} = fir.embox %[[INIT_ARG1]]
! CHECK:   %{{.*}} = fir.embox %[[INIT_ARG0]]
! CHECK:   %{{.*}}:2 = hlfir.declare %[[INIT_ARG0]] {uniq_name = "omp_orig"}
! CHECK:   %{{.*}}:2 = hlfir.declare %[[INIT_ARG1]] {uniq_name = "omp_priv"}
! CHECK:   omp.yield(%[[INIT_ARG1]] : !fir.ref<[[TY]]>)
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<[[TY]]>, %[[ARG1:.*]]: !fir.ref<[[TY]]>):
! CHECK:   %[[OMP_IN:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "omp_in"}
! CHECK:   %[[OMP_OUT:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "omp_out"}
! CHECK:   %[[OUT_X:.*]] = hlfir.designate %[[OMP_OUT]]#0{"x"} : (!fir.ref<[[TY]]>) -> !fir.ref<i32>
! CHECK:   %[[OUT_X_VAL:.*]] = fir.load %[[OUT_X]] : !fir.ref<i32>
! CHECK:   %[[IN_X:.*]] = hlfir.designate %[[OMP_IN]]#0{"x"} : (!fir.ref<[[TY]]>) -> !fir.ref<i32>
! CHECK:   %[[IN_X_VAL:.*]] = fir.load %[[IN_X]] : !fir.ref<i32>
! CHECK:   %[[ADD:.*]] = arith.addi %[[OUT_X_VAL]], %[[IN_X_VAL]] : i32
! CHECK:   %[[OUT_X2:.*]] = hlfir.designate %[[OMP_OUT]]#0{"x"} : (!fir.ref<[[TY]]>) -> !fir.ref<i32>
! CHECK:   hlfir.assign %[[ADD]] to %[[OUT_X2]] : i32, !fir.ref<i32>
! CHECK:   omp.yield(%[[ARG0]] : !fir.ref<[[TY]]>)
! CHECK: }
