! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine target_nested_teams_distribute_private()
  integer :: i, k
  !$omp target
    !$omp teams distribute private(k)
    do i = 1, 10
      k = 10
    end do
    !$omp end teams distribute
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_nested_teams_distribute_private
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.map.info {{.*}} name("i")
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFtarget_nested_teams_distribute_privateEk"}
! CHECK: omp.teams
! CHECK: omp.distribute private({{.*}}@_QFtarget_nested_teams_distribute_privateEk_private_i32

subroutine target_nested_teams_distribute_private_thread_limit(k)
  integer :: i, k
  !$omp target
    !$omp teams distribute private(k) thread_limit(k)
    do i = 1, 10
      k = 10
    end do
    !$omp end teams distribute
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_nested_teams_distribute_private_thread_limit
! CHECK: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: omp.teams thread_limit
! CHECK: omp.distribute private({{.*}}@_QFtarget_nested_teams_distribute_private_thread_limitEk_private_i32

subroutine target_nested_parallel_private()
  integer :: k
  !$omp target
    !$omp parallel private(k)
      k = 10
    !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_nested_parallel_private
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: fir.alloca i32 {bindc_name = "k", pinned, uniq_name = "_QFtarget_nested_parallel_privateEk"}
! CHECK: omp.parallel private({{.*}}@_QFtarget_nested_parallel_privateEk_private_i32

subroutine target_nested_parallel_private_num_threads(k)
  integer :: k
  !$omp target
    !$omp parallel private(k) num_threads(k)
      k = 10
    !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_nested_parallel_private_num_threads
! CHECK: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: omp.parallel {{.*}}private({{.*}}@_QFtarget_nested_parallel_private_num_threadsEk_private_i32
