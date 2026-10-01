! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine target_parallel_do_private_num_threads(n, out)
  integer :: n, out, i
  out = 0
  !$omp target parallel do private(n) num_threads(n) map(tofrom: out)
  do i = 1, 4
    n = i
    out = out + n
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do_private_num_threads
! CHECK: %[[N_VAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[OUT_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("out")
! CHECK: %[[N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: omp.target kernel_type(spmd) host_eval({{.*}}, %[[N_VAL]] -> %[[N_HOST_ARG:arg[0-9]+]] : {{.*}}) map_entries(%[[OUT_MAP]] -> %{{.*}}, %[[N_MAP]] -> %[[N_MAP_ARG:arg[0-9]+]], %[[I_MAP]] -> %{{.*}} : {{.*}}) {
! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %[[N_MAP_ARG]] {uniq_name = "_QFtarget_parallel_do_private_num_threadsEn"}
! CHECK: omp.parallel num_threads(%[[N_HOST_ARG]] : i32) {
! CHECK: omp.wsloop private({{.*}}@_QFtarget_parallel_do_private_num_threadsEn_private_i32 %[[N_DECL]]#0 -> %arg{{[0-9]+}}

subroutine target_teams_distribute_private_thread_limit(n, out)
  integer :: n, out, i
  out = 0
  !$omp target teams distribute private(n) thread_limit(n) map(tofrom: out)
  do i = 1, 4
    n = i
    out = out + n
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_private_thread_limit
! CHECK: %[[OUT_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("out")
! CHECK: %[[N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: omp.target kernel_type(generic) host_eval({{.*}}) map_entries(%[[OUT_MAP]] -> %{{.*}}, %[[N_MAP]] -> %[[N_MAP_ARG:arg[0-9]+]], %[[I_MAP]] -> %{{.*}} : {{.*}}) {
! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %[[N_MAP_ARG]] {uniq_name = "_QFtarget_teams_distribute_private_thread_limitEn"}
! CHECK: %[[N_LOAD:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>
! CHECK: omp.teams thread_limit(%[[N_LOAD]] : i32) {
! CHECK: omp.distribute private({{.*}}@_QFtarget_teams_distribute_private_thread_limitEn_private_i32 %[[N_DECL]]#0 -> %arg{{[0-9]+}}

subroutine target_teams_distribute_private_num_teams(n, out)
  integer :: n, out, i
  out = 0
  !$omp target teams distribute private(n) num_teams(n) map(tofrom: out)
  do i = 1, 4
    n = i
    out = out + n
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_private_num_teams
! CHECK: %[[N_VAL:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %[[OUT_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("out")
! CHECK: %[[N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: omp.target kernel_type(generic) host_eval({{.*}}, %[[N_VAL]] -> %[[N_HOST_ARG:arg[0-9]+]] : {{.*}}) map_entries(%[[OUT_MAP]] -> %{{.*}}, %[[N_MAP]] -> %[[N_MAP_ARG:arg[0-9]+]], %[[I_MAP]] -> %{{.*}} : {{.*}}) {
! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %[[N_MAP_ARG]] {uniq_name = "_QFtarget_teams_distribute_private_num_teamsEn"}
! CHECK: omp.teams num_teams( to %[[N_HOST_ARG]] : i32) {
! CHECK: omp.distribute private({{.*}}@_QFtarget_teams_distribute_private_num_teamsEn_private_i32 %[[N_DECL]]#0 -> %arg{{[0-9]+}}
