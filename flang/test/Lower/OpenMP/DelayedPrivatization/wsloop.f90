! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization-staging \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization-staging -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine wsloop_private
    implicit none
    integer :: x, i

    !$omp parallel do firstprivate(x)
    do i = 0, 10
      x = x + i
    end do
end subroutine wsloop_private

! CHECK: omp.private {type = private} @[[I_PRIVATIZER:.*i_private_i32]]
! CHECK: omp.private {type = firstprivate} @[[X_PRIVATIZER:.*x_firstprivate_i32]]

! CHECK: func.func @{{.*}}() {
! CHECK:   %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}i"}
! CHECK:   %[[X_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}x"}

! CHECK:   omp.parallel {
! CHECK:     omp.wsloop private(
! CHECK-SAME:  @[[X_PRIVATIZER]] %[[X_DECL]]#0 -> %[[X_ARG:[^[:space:]]+]],
! CHECK-SAME:  @[[I_PRIVATIZER]] %[[I_DECL]]#0 -> %[[I_ARG:.*]] : {{.*}}) {

! CHECK:       omp.loop_nest (%[[IV:.*]]) : i32 = {{.*}} {
! CHECK:         %[[X_PRIV_DECL:.*]]:2 = hlfir.declare %[[X_ARG]] {uniq_name = "{{.*}}x"}
! CHECK:         %[[I_PRIV_DECL:.*]]:2 = hlfir.declare %[[I_ARG]] {uniq_name = "{{.*}}i"}
! CHECK:         fir.store %[[IV]] to %[[I_PRIV_DECL]]#1
! CHECK:         %[[X_VAL:.*]] = fir.load %[[X_PRIV_DECL]]#0
! CHECK:         %[[I_VAL:.*]] = fir.load %[[I_PRIV_DECL]]#0
! CHECK:         %[[ADD_VAL:.*]] = arith.addi %[[X_VAL]], %[[I_VAL]]
! CHECK:         hlfir.assign %[[ADD_VAL]] to %[[X_PRIV_DECL]]#0
! CHECK:         omp.yield
! CHECK:       }
! CHECK:     }

! CHECK:     omp.terminator
! CHECK:   }
! CHECK: }
