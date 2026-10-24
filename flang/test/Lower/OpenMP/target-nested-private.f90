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
! CHECK: fir.alloca i32 <{bindc_name = "k", pinned, uniq_name = "_QFtarget_nested_teams_distribute_privateEk"}>
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
! CHECK: fir.alloca i32 <{bindc_name = "k", pinned, uniq_name = "_QFtarget_nested_parallel_privateEk"}>
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

program main
  implicit none
  integer :: i, n

  !$omp target
    !$omp teams
      !$omp distribute private(i)
      do i = 1, n
      end do
    !$omp end teams
  !$omp end target
end program

! CHECK-LABEL: func.func @_QQmain
! CHECK-NOT: omp.map.info {{.*}} name("i")
! CHECK: omp.map.info {{.*}} name("n")
! CHECK-NOT: omp.map.info {{.*}} name("i")
! CHECK: omp.target
! CHECK: %[[I_ALLOCA:.*]] = fir.alloca i32 <{bindc_name = "i", pinned, uniq_name = "_QFEi"}>
! CHECK: %[[I_DECL:.*]]:2 = hlfir.declare %[[I_ALLOCA]] uniq_name("_QFEi")
! CHECK: omp.teams
! CHECK: omp.distribute private({{.*}}@_QFEi_private_i32 %[[I_DECL]]#0 -> %[[I_PRIV_ARG:.*]] : !fir.ref<i32>)
! CHECK: omp.loop_nest
! CHECK: hlfir.declare %[[I_PRIV_ARG]] uniq_name("_QFEi")

subroutine target_teams_distribute_nested_parallel_private(n)
  implicit none
  integer :: i, n

  !$omp target teams
    !$omp distribute
    do i = 1, n
      !$omp parallel private(i)
      !$omp end parallel
    end do
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_nested_parallel_private
! CHECK: %[[I_MAP:.*]] = omp.map.info {{.*}} name("i")
! CHECK: %[[N_MAP:.*]] = omp.map.info {{.*}} name("n")
! CHECK: omp.target {{.*}}map_entries(%[[I_MAP]] -> %[[I_MAP_ARG:.*]], %[[N_MAP]] -> %{{.*}} : !fir.ref<i32>, !fir.ref<i32>)
! CHECK-NEXT: %[[I_TARGET_DECL:.*]]:2 = hlfir.declare %[[I_MAP_ARG]] uniq_name("_QFtarget_teams_distribute_nested_parallel_privateEi")
! CHECK-NOT: fir.alloca i32 <{bindc_name = "i", pinned, uniq_name = "_QFtarget_teams_distribute_nested_parallel_privateEi"}>
! CHECK: omp.teams
! CHECK: omp.distribute private({{.*}}@_QFtarget_teams_distribute_nested_parallel_privateEi_private_i32 %[[I_TARGET_DECL]]#0 -> %[[I_DIST_ARG:.*]] : !fir.ref<i32>)
! CHECK: omp.loop_nest
! CHECK: hlfir.declare %[[I_DIST_ARG]] uniq_name("_QFtarget_teams_distribute_nested_parallel_privateEi")
! CHECK: omp.parallel private({{.*}}@_QFtarget_teams_distribute_nested_parallel_privateEi_private_i32 {{.*}} -> %[[I_PAR_ARG:.*]] : !fir.ref<i32>)
! CHECK: hlfir.declare %[[I_PAR_ARG]] uniq_name("_QFtarget_teams_distribute_nested_parallel_privateEi")
