! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: omp.declare_reduction @[[RED:.*]] : i32 init {

! CHECK: func.func @_QPreduction_teams() {
subroutine reduction_teams()
  integer :: i
  i = 0

  ! CHECK: omp.teams reduction(@[[RED]] %{{.*}}#0 -> %[[PRIV_I:.*]] : !fir.ref<i32>) {
  !$omp teams reduction(+:i)
    ! CHECK: %[[DECL_I:.*]]:2 = hlfir.declare %[[PRIV_I]]
    ! CHECK: %{{.*}} = fir.load %[[DECL_I]]#0 : !fir.ref<i32>
    ! CHECK: hlfir.assign %{{.*}} to %[[DECL_I]]#0 : i32, !fir.ref<i32>
    i = i + 1
  !$omp end teams
end subroutine reduction_teams
