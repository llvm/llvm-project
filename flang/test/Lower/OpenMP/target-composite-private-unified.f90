!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module types
  type :: dt
    integer :: a
    real :: b
  end type
  type :: dt_alloc
    integer, allocatable :: a
  end type
  type :: dt_alloc_arr
    integer, allocatable :: a(:)
  end type
end module

subroutine target_teams(n, assumed)
  use types
  integer :: n, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target teams private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  s = 1
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[TEAMS_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TEAMS_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TEAMS_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[TEAMS_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[TEAMS_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[TEAMS_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TEAMS_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[TEAMS_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[TEAMS_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[TEAMS_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TEAMS_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TEAMS_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[TEAMS_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[TEAMS_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[TEAMS_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[TEAMS_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(generic) map_entries(%[[TEAMS_ALLOC_MAP]] -> %{{.*}}, %[[TEAMS_DA_MAP]] -> %{{.*}}, %[[TEAMS_ASSUMED_MAP]] -> %{{.*}}, %[[TEAMS_ALLOC_ARR_MAP]] -> %{{.*}}, %[[TEAMS_DAA_MAP]] -> %{{.*}}, %[[TEAMS_AUTO_ARR_MAP]] -> %{{.*}}, %[[TEAMS_N_MAP]] -> %{{.*}}, %[[TEAMS_ALLOC_ATTACH]] -> %{{.*}}, %[[TEAMS_ASSUMED_ATTACH]] -> %{{.*}}, %[[TEAMS_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[TEAMS_ALLOC_MEMBER]] -> %{{.*}}, %[[TEAMS_ASSUMED_MEMBER]] -> %{{.*}}, %[[TEAMS_ALLOC_ARR_MEMBER]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teamsEalloc"}
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teamsEda"}
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teamsEassumed"}
! CHECK: hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teamsEalloc_arr"}
! CHECK: hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teamsEdaa"}
! CHECK: hlfir.declare %arg{{[0-9]+}}(%{{.*}}) {uniq_name = "_QFtarget_teamsEauto_arr"}
! CHECK: omp.teams {
! CHECK: fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFtarget_teamsEs"}
! CHECK: fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_teamsEarr"}
! CHECK: fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "alloc", pinned, uniq_name = "_QFtarget_teamsEalloc"}
! CHECK: fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFtarget_teamsEptr"}
! CHECK: fir.alloca !fir.type<_QMtypesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFtarget_teamsEd"}
! CHECK: fir.alloca !fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}> {bindc_name = "da", pinned, uniq_name = "_QFtarget_teamsEda"}
! CHECK: fir.alloca !fir.array<?xi32>, %{{.*}} {bindc_name = "assumed", pinned, uniq_name = "_QFtarget_teamsEassumed"}
! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "alloc_arr", pinned, uniq_name = "_QFtarget_teamsEalloc_arr"}
! CHECK: fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFtarget_teamsEptr_arr"}
! CHECK: fir.alloca !fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {bindc_name = "daa", pinned, uniq_name = "_QFtarget_teamsEdaa"}
! CHECK: fir.alloca !fir.array<?xi32>, %{{.*}} {bindc_name = "auto_arr", pinned, uniq_name = "_QFtarget_teamsEauto_arr"}

subroutine target_parallel(n, assumed)
  use types
  integer :: n, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target parallel private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  s = 1
  !$omp end target parallel
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[PAR_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PAR_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[PAR_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[PAR_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[PAR_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[PAR_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PAR_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[PAR_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[PAR_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[PAR_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PAR_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[PAR_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[PAR_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[PAR_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[PAR_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[PAR_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(generic) map_entries(%[[PAR_ALLOC_MAP]] -> %{{.*}}, %[[PAR_DA_MAP]] -> %{{.*}}, %[[PAR_ASSUMED_MAP]] -> %{{.*}}, %[[PAR_ALLOC_ARR_MAP]] -> %{{.*}}, %[[PAR_DAA_MAP]] -> %{{.*}}, %[[PAR_AUTO_ARR_MAP]] -> %{{.*}}, %[[PAR_N_MAP]] -> %{{.*}}, %[[PAR_ALLOC_ATTACH]] -> %{{.*}}, %[[PAR_ASSUMED_ATTACH]] -> %{{.*}}, %[[PAR_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[PAR_ALLOC_MEMBER]] -> %{{.*}}, %[[PAR_ASSUMED_MEMBER]] -> %{{.*}}, %[[PAR_ALLOC_ARR_MEMBER]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: %[[PAR_ALLOC_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_parallelEalloc"}
! CHECK: %[[PAR_DA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallelEda"}
! CHECK: %[[PAR_ASSUMED_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallelEassumed"}
! CHECK: %[[PAR_ALLOC_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_parallelEalloc_arr"}
! CHECK: %[[PAR_DAA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallelEdaa"}
! CHECK: %[[PAR_AUTO_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}}(%{{.*}}) {uniq_name = "_QFtarget_parallelEauto_arr"}
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "da", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "assumed", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc_arr", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "daa", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "auto_arr", pinned
! CHECK: %[[PAR_S_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFtarget_parallelEs"}
! CHECK: %[[PAR_S_DECL:.*]]:2 = hlfir.declare %[[PAR_S_ALLOCA]]
! CHECK: %[[PAR_ARR_ALLOCA:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_parallelEarr"}
! CHECK: %[[PAR_ARR_DECL:.*]]:2 = hlfir.declare %[[PAR_ARR_ALLOCA]]
! CHECK: %[[PAR_PTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFtarget_parallelEptr"}
! CHECK: %[[PAR_PTR_DECL:.*]]:2 = hlfir.declare %[[PAR_PTR_ALLOCA]]
! CHECK: %[[PAR_D_ALLOCA:.*]] = fir.alloca !fir.type<_QMtypesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFtarget_parallelEd"}
! CHECK: %[[PAR_D_DECL:.*]]:2 = hlfir.declare %[[PAR_D_ALLOCA]]
! CHECK: %[[PAR_PTR_ARR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFtarget_parallelEptr_arr"}
! CHECK: %[[PAR_PTR_ARR_DECL:.*]]:2 = hlfir.declare %[[PAR_PTR_ARR_ALLOCA]]
! CHECK: fir.store %[[PAR_ASSUMED_DECL]]#0 to %[[PAR_ASSUMED_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: fir.store %[[PAR_AUTO_ARR_DECL]]#0 to %[[PAR_AUTO_ARR_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: omp.parallel private(
! CHECK-SAME: @_QFtarget_parallelEs_private_i32 %[[PAR_S_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEarr_private_10xi32 %[[PAR_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEalloc_private_box_heap_i32 %[[PAR_ALLOC_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEptr_private_box_ptr_i32 %[[PAR_PTR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEd_private_rec__QMtypesTdt %[[PAR_D_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEda_private_rec__QMtypesTdt_alloc %[[PAR_DA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEassumed_private_box_Uxi32 %[[PAR_ASSUMED_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEalloc_arr_private_box_heap_Uxi32 %[[PAR_ALLOC_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEptr_arr_private_box_ptr_Uxi32 %[[PAR_PTR_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEdaa_private_rec__QMtypesTdt_alloc_arr %[[PAR_DAA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallelEauto_arr_private_box_Uxi32 %[[PAR_AUTO_ARR_COPY]] -> %arg{{[0-9]+}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>) {

subroutine target_teams_distribute(n, assumed)
  use types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target teams distribute private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  do i = 1, n
    s = i
  end do
  !$omp end target teams distribute
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[TTD_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTD_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TTD_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[TTD_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[TTD_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[TTD_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTD_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[TTD_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[TTD_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[TTD_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTD_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TTD_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[TTD_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[TTD_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[TTD_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[TTD_I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: %[[TTD_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[TTD_BOUND_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(generic) host_eval({{.*}}) map_entries(%[[TTD_ALLOC_MAP]] -> %{{.*}}, %[[TTD_DA_MAP]] -> %{{.*}}, %[[TTD_ASSUMED_MAP]] -> %{{.*}}, %[[TTD_ALLOC_ARR_MAP]] -> %{{.*}}, %[[TTD_DAA_MAP]] -> %{{.*}}, %[[TTD_AUTO_ARR_MAP]] -> %{{.*}}, %[[TTD_I_MAP]] -> %{{.*}}, %[[TTD_N_MAP]] -> %{{.*}}, %[[TTD_BOUND_N_MAP]] -> %{{.*}}, %[[TTD_ALLOC_ATTACH]] -> %{{.*}}, %[[TTD_ASSUMED_ATTACH]] -> %{{.*}}, %[[TTD_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[TTD_ALLOC_MEMBER]] -> %{{.*}}, %[[TTD_ASSUMED_MEMBER]] -> %{{.*}}, %[[TTD_ALLOC_ARR_MEMBER]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: %[[TTD_ALLOC_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teams_distributeEalloc"}
! CHECK: %[[TTD_DA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distributeEda"}
! CHECK: %[[TTD_ASSUMED_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distributeEassumed"}
! CHECK: %[[TTD_ALLOC_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teams_distributeEalloc_arr"}
! CHECK: %[[TTD_DAA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distributeEdaa"}
! CHECK: %[[TTD_AUTO_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}}(%{{.*}}) {uniq_name = "_QFtarget_teams_distributeEauto_arr"}
! CHECK: %[[TTD_I_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distributeEi"}
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "da", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "assumed", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc_arr", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "daa", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "auto_arr", pinned
! CHECK: %[[TTD_S_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFtarget_teams_distributeEs"}
! CHECK: %[[TTD_S_DECL:.*]]:2 = hlfir.declare %[[TTD_S_ALLOCA]]
! CHECK: %[[TTD_ARR_ALLOCA:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_teams_distributeEarr"}
! CHECK: %[[TTD_ARR_DECL:.*]]:2 = hlfir.declare %[[TTD_ARR_ALLOCA]]
! CHECK: %[[TTD_PTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFtarget_teams_distributeEptr"}
! CHECK: %[[TTD_PTR_DECL:.*]]:2 = hlfir.declare %[[TTD_PTR_ALLOCA]]
! CHECK: %[[TTD_D_ALLOCA:.*]] = fir.alloca !fir.type<_QMtypesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFtarget_teams_distributeEd"}
! CHECK: %[[TTD_D_DECL:.*]]:2 = hlfir.declare %[[TTD_D_ALLOCA]]
! CHECK: %[[TTD_PTR_ARR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFtarget_teams_distributeEptr_arr"}
! CHECK: %[[TTD_PTR_ARR_DECL:.*]]:2 = hlfir.declare %[[TTD_PTR_ARR_ALLOCA]]
! CHECK: omp.teams {
! CHECK: fir.store %[[TTD_ASSUMED_DECL]]#0 to %[[TTD_ASSUMED_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: fir.store %[[TTD_AUTO_ARR_DECL]]#0 to %[[TTD_AUTO_ARR_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: omp.distribute private(
! CHECK-SAME: @_QFtarget_teams_distributeEs_private_i32 %[[TTD_S_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEarr_private_10xi32 %[[TTD_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEalloc_private_box_heap_i32 %[[TTD_ALLOC_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEptr_private_box_ptr_i32 %[[TTD_PTR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEd_private_rec__QMtypesTdt %[[TTD_D_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEda_private_rec__QMtypesTdt_alloc %[[TTD_DA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEassumed_private_box_Uxi32 %[[TTD_ASSUMED_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEalloc_arr_private_box_heap_Uxi32 %[[TTD_ALLOC_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEptr_arr_private_box_ptr_Uxi32 %[[TTD_PTR_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEdaa_private_rec__QMtypesTdt_alloc_arr %[[TTD_DAA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEauto_arr_private_box_Uxi32 %[[TTD_AUTO_ARR_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distributeEi_private_i32 %[[TTD_I_DECL]]#0 -> %arg{{[0-9]+}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<i32>) {

subroutine target_only(n, assumed)
  use types
  integer :: n, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  s = 1
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_only
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[ONLY_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32) name("")
! CHECK: %[[ONLY_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} members(%[[ONLY_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>)
! CHECK: %[[ONLY_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32)
! CHECK: %[[ONLY_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) mapper(@_QMtypesTdt_alloc_omp_default_mapper)
! CHECK: %[[ONLY_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("")
! CHECK: %[[ONLY_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} members(%[[ONLY_ASSUMED_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
! CHECK: %[[ONLY_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32)
! CHECK: %[[ONLY_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("")
! CHECK: %[[ONLY_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} members(%[[ONLY_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
! CHECK: %[[ONLY_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32)
! CHECK: %[[ONLY_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(tofrom) capture(ByRef) mapper(@_QMtypesTdt_alloc_arr_omp_default_mapper)
! CHECK: %[[ONLY_AUTO_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(tofrom) {{.*}} name("")
! CHECK: %[[ONLY_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, to) {{.*}} members(%[[ONLY_AUTO_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
! CHECK: %[[ONLY_AUTO_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32)
! CHECK: omp.target kernel_type(generic) map_entries(%[[ONLY_ALLOC_MAP]] -> %{{.*}}, %[[ONLY_DA_MAP]] -> %{{.*}}, %[[ONLY_ASSUMED_MAP]] -> %{{.*}}, %[[ONLY_ALLOC_ARR_MAP]] -> %{{.*}}, %[[ONLY_DAA_MAP]] -> %{{.*}}, %[[ONLY_AUTO_ARR_MAP]] -> %{{.*}}, %[[ONLY_ALLOC_ATTACH]] -> %{{.*}}, %[[ONLY_ASSUMED_ATTACH]] -> %{{.*}}, %[[ONLY_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[ONLY_AUTO_ARR_ATTACH]] -> %{{.*}}, %[[ONLY_ALLOC_MEMBER]] -> %{{.*}}, %[[ONLY_ASSUMED_MEMBER]] -> %{{.*}}, %[[ONLY_ALLOC_ARR_MEMBER]] -> %{{.*}}, %[[ONLY_AUTO_ARR_MEMBER]] -> %{{.*}} : {{.*}}) private(
! CHECK-SAME: @_QFtarget_onlyEs_private_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_onlyEarr_private_10xi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_onlyEalloc_private_box_heap_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}} [map_idx=0],
! CHECK-SAME: @_QFtarget_onlyEptr_private_box_ptr_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_onlyEd_private_rec__QMtypesTdt %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_onlyEda_private_rec__QMtypesTdt_alloc %{{[0-9]+}}#0 -> %arg{{[0-9]+}} [map_idx=1],
! CHECK-SAME: @_QFtarget_onlyEassumed_private_box_Uxi32 %{{[0-9]+}} -> %arg{{[0-9]+}} [map_idx=2],
! CHECK-SAME: @_QFtarget_onlyEalloc_arr_private_box_heap_Uxi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}} [map_idx=3],
! CHECK-SAME: @_QFtarget_onlyEptr_arr_private_box_ptr_Uxi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_onlyEdaa_private_rec__QMtypesTdt_alloc_arr %{{[0-9]+}}#0 -> %arg{{[0-9]+}} [map_idx=4],
! CHECK-SAME: @_QFtarget_onlyEauto_arr_private_box_Uxi32 %{{[0-9]+}} -> %arg{{[0-9]+}} [map_idx=5] : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>) {

subroutine target_nested(n, assumed)
  use types
  integer :: n, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target
  !$omp parallel private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  s = 1
  !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPtarget_nested
! CHECK: %[[NEST_S_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("s")
! CHECK: %[[NEST_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("arr")
! CHECK: %[[NEST_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[NEST_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[NEST_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[NEST_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[NEST_PTR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[NEST_PTR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[NEST_PTR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("ptr")
! CHECK: %[[NEST_PTR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("ptr")
! CHECK: %[[NEST_D_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("d")
! CHECK: %[[NEST_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[NEST_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[NEST_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[NEST_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[NEST_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[NEST_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[NEST_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[NEST_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[NEST_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[NEST_PTR_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[NEST_PTR_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[NEST_PTR_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("ptr_arr")
! CHECK: %[[NEST_PTR_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("ptr_arr")
! CHECK: %[[NEST_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[NEST_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[NEST_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(generic) map_entries(%[[NEST_S_MAP]] -> %{{.*}}, %[[NEST_ARR_MAP]] -> %{{.*}}, %[[NEST_ALLOC_MAP]] -> %{{.*}}, %[[NEST_PTR_MAP]] -> %{{.*}}, %[[NEST_D_MAP]] -> %{{.*}}, %[[NEST_DA_MAP]] -> %{{.*}}, %[[NEST_ASSUMED_MAP]] -> %{{.*}}, %[[NEST_ALLOC_ARR_MAP]] -> %{{.*}}, %[[NEST_PTR_ARR_MAP]] -> %{{.*}}, %[[NEST_DAA_MAP]] -> %{{.*}}, %[[NEST_AUTO_ARR_MAP]] -> %{{.*}}, %[[NEST_N_MAP]] -> %{{.*}}, %[[NEST_ALLOC_ATTACH]] -> %{{.*}}, %[[NEST_PTR_ATTACH]] -> %{{.*}}, %[[NEST_ASSUMED_ATTACH]] -> %{{.*}}, %[[NEST_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[NEST_PTR_ARR_ATTACH]] -> %{{.*}}, %[[NEST_ALLOC_MEMBER]] -> %{{.*}}, %[[NEST_PTR_MEMBER]] -> %{{.*}}, %[[NEST_ASSUMED_MEMBER]] -> %{{.*}}, %[[NEST_ALLOC_ARR_MEMBER]] -> %{{.*}}, %[[NEST_PTR_ARR_MEMBER]] -> %{{.*}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: omp.parallel private(
! CHECK-SAME: @_QFtarget_nestedEs_private_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEarr_private_10xi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEalloc_private_box_heap_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEptr_private_box_ptr_i32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEd_private_rec__QMtypesTdt %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEda_private_rec__QMtypesTdt_alloc %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEassumed_private_box_Uxi32 %{{[0-9]+}} -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEalloc_arr_private_box_heap_Uxi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEptr_arr_private_box_ptr_Uxi32 %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEdaa_private_rec__QMtypesTdt_alloc_arr %{{[0-9]+}}#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_nestedEauto_arr_private_box_Uxi32 %{{[0-9]+}} -> %arg{{[0-9]+}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>) {

subroutine target_teams_distribute_simd(n, assumed)
  use types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target teams distribute simd private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  do i = 1, n
    s = i
  end do
  !$omp end target teams distribute simd
end subroutine

! CHECK-LABEL: func.func @_QPtarget_teams_distribute_simd
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[TTDS_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTDS_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TTDS_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[TTDS_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[TTDS_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[TTDS_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTDS_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[TTDS_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[TTDS_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[TTDS_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[TTDS_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[TTDS_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[TTDS_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[TTDS_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[TTDS_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[TTDS_I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: %[[TTDS_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[TTDS_BOUND_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(generic) host_eval({{.*}}) map_entries(%[[TTDS_ALLOC_MAP]] -> %{{.*}}, %[[TTDS_DA_MAP]] -> %{{.*}}, %[[TTDS_ASSUMED_MAP]] -> %{{.*}}, %[[TTDS_ALLOC_ARR_MAP]] -> %{{.*}}, %[[TTDS_DAA_MAP]] -> %{{.*}}, %[[TTDS_AUTO_ARR_MAP]] -> %{{.*}}, %[[TTDS_I_MAP]] -> %{{.*}}, %[[TTDS_N_MAP]] -> %{{.*}}, %[[TTDS_BOUND_N_MAP]] -> %{{.*}}, %[[TTDS_ALLOC_ATTACH]] -> %{{.*}}, %[[TTDS_ASSUMED_ATTACH]] -> %{{.*}}, %[[TTDS_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[TTDS_ALLOC_MEMBER]] -> %{{.*}}, %[[TTDS_ASSUMED_MEMBER]] -> %{{.*}}, %[[TTDS_ALLOC_ARR_MEMBER]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: %[[TTDS_ALLOC_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teams_distribute_simdEalloc"}
! CHECK: %[[TTDS_DA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distribute_simdEda"}
! CHECK: %[[TTDS_ASSUMED_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distribute_simdEassumed"}
! CHECK: %[[TTDS_ALLOC_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_teams_distribute_simdEalloc_arr"}
! CHECK: %[[TTDS_DAA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distribute_simdEdaa"}
! CHECK: %[[TTDS_AUTO_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}}(%{{.*}}) {uniq_name = "_QFtarget_teams_distribute_simdEauto_arr"}
! CHECK: %[[TTDS_I_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_teams_distribute_simdEi"}
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "da", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "assumed", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc_arr", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "daa", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "auto_arr", pinned
! CHECK: %[[TTDS_S_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFtarget_teams_distribute_simdEs"}
! CHECK: %[[TTDS_S_DECL:.*]]:2 = hlfir.declare %[[TTDS_S_ALLOCA]]
! CHECK: %[[TTDS_ARR_ALLOCA:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_teams_distribute_simdEarr"}
! CHECK: %[[TTDS_ARR_DECL:.*]]:2 = hlfir.declare %[[TTDS_ARR_ALLOCA]]
! CHECK: %[[TTDS_PTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFtarget_teams_distribute_simdEptr"}
! CHECK: %[[TTDS_PTR_DECL:.*]]:2 = hlfir.declare %[[TTDS_PTR_ALLOCA]]
! CHECK: %[[TTDS_D_ALLOCA:.*]] = fir.alloca !fir.type<_QMtypesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFtarget_teams_distribute_simdEd"}
! CHECK: %[[TTDS_D_DECL:.*]]:2 = hlfir.declare %[[TTDS_D_ALLOCA]]
! CHECK: %[[TTDS_PTR_ARR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFtarget_teams_distribute_simdEptr_arr"}
! CHECK: %[[TTDS_PTR_ARR_DECL:.*]]:2 = hlfir.declare %[[TTDS_PTR_ARR_ALLOCA]]
! CHECK: omp.teams {
! CHECK: fir.store %[[TTDS_ASSUMED_DECL]]#0 to %[[TTDS_ASSUMED_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: fir.store %[[TTDS_AUTO_ARR_DECL]]#0 to %[[TTDS_AUTO_ARR_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: omp.distribute {
! CHECK: omp.simd private(
! CHECK-SAME: @_QFtarget_teams_distribute_simdEs_private_i32 %[[TTDS_S_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEarr_private_10xi32 %[[TTDS_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEalloc_private_box_heap_i32 %[[TTDS_ALLOC_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEptr_private_box_ptr_i32 %[[TTDS_PTR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEd_private_rec__QMtypesTdt %[[TTDS_D_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEda_private_rec__QMtypesTdt_alloc %[[TTDS_DA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEassumed_private_box_Uxi32 %[[TTDS_ASSUMED_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEalloc_arr_private_box_heap_Uxi32 %[[TTDS_ALLOC_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEptr_arr_private_box_ptr_Uxi32 %[[TTDS_PTR_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEdaa_private_rec__QMtypesTdt_alloc_arr %[[TTDS_DAA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEauto_arr_private_box_Uxi32 %[[TTDS_AUTO_ARR_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_teams_distribute_simdEi_private_i32 %[[TTDS_I_DECL]]#0 -> %arg{{[0-9]+}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<i32>) {

subroutine target_parallel_do(n, assumed)
  use types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target parallel do private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
  do i = 1, n
    s = i
  end do
  !$omp end target parallel do
end subroutine

! CHECK-LABEL: func.func @_QPtarget_parallel_do
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: %[[PDO_ALLOC_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PDO_ALLOC_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[PDO_ALLOC_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) name("alloc")
! CHECK: %[[PDO_ALLOC_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc")
! CHECK: %[[PDO_DA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("da")
! CHECK: %[[PDO_ASSUMED_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PDO_ASSUMED_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, target_param, private, attach) {{.*}} members(%[[PDO_ASSUMED_MEMBER]] {{.*}}) name("assumed")
! CHECK: %[[PDO_ASSUMED_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("assumed")
! CHECK: %[[PDO_ALLOC_ARR_MEMBER:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("")
! CHECK: %[[PDO_ALLOC_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(always, implicit, to) {{.*}} members(%[[PDO_ALLOC_ARR_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("alloc_arr")
! CHECK: %[[PDO_ALLOC_ARR_ATTACH:.*]] = omp.map.info {{.*}} map_clauses(attach, ref_ptr, ref_ptee) {{.*}} name("alloc_arr")
! CHECK: %[[PDO_DAA_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("daa")
! CHECK: %[[PDO_AUTO_ARR_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit, tofrom) {{.*}} name("auto_arr")
! CHECK: %[[PDO_I_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("i")
! CHECK: %[[PDO_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("n")
! CHECK: %[[PDO_BOUND_N_MAP:.*]] = omp.map.info {{.*}} map_clauses(implicit) capture(ByCopy) name("")
! CHECK: omp.target kernel_type(spmd) host_eval({{.*}}) map_entries(%[[PDO_ALLOC_MAP]] -> %{{.*}}, %[[PDO_DA_MAP]] -> %{{.*}}, %[[PDO_ASSUMED_MAP]] -> %{{.*}}, %[[PDO_ALLOC_ARR_MAP]] -> %{{.*}}, %[[PDO_DAA_MAP]] -> %{{.*}}, %[[PDO_AUTO_ARR_MAP]] -> %{{.*}}, %[[PDO_I_MAP]] -> %{{.*}}, %[[PDO_N_MAP]] -> %{{.*}}, %[[PDO_BOUND_N_MAP]] -> %{{.*}}, %[[PDO_ALLOC_ATTACH]] -> %{{.*}}, %[[PDO_ASSUMED_ATTACH]] -> %{{.*}}, %[[PDO_ALLOC_ARR_ATTACH]] -> %{{.*}}, %[[PDO_ALLOC_MEMBER]] -> %{{.*}}, %[[PDO_ASSUMED_MEMBER]] -> %{{.*}}, %[[PDO_ALLOC_ARR_MEMBER]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<i32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
! CHECK: %[[PDO_ALLOC_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_parallel_doEalloc"}
! CHECK: %[[PDO_DA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallel_doEda"}
! CHECK: %[[PDO_ASSUMED_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallel_doEassumed"}
! CHECK: %[[PDO_ALLOC_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtarget_parallel_doEalloc_arr"}
! CHECK: %[[PDO_DAA_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallel_doEdaa"}
! CHECK: %[[PDO_AUTO_ARR_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}}(%{{.*}}) {uniq_name = "_QFtarget_parallel_doEauto_arr"}
! CHECK: %[[PDO_I_DECL:.*]]:2 = hlfir.declare %arg{{[0-9]+}} {uniq_name = "_QFtarget_parallel_doEi"}
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "da", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "assumed", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "alloc_arr", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "daa", pinned
! CHECK-NOT: fir.alloca {{.*}} {bindc_name = "auto_arr", pinned
! CHECK: %[[PDO_S_ALLOCA:.*]] = fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFtarget_parallel_doEs"}
! CHECK: %[[PDO_S_DECL:.*]]:2 = hlfir.declare %[[PDO_S_ALLOCA]]
! CHECK: %[[PDO_ARR_ALLOCA:.*]] = fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFtarget_parallel_doEarr"}
! CHECK: %[[PDO_ARR_DECL:.*]]:2 = hlfir.declare %[[PDO_ARR_ALLOCA]]
! CHECK: %[[PDO_PTR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFtarget_parallel_doEptr"}
! CHECK: %[[PDO_PTR_DECL:.*]]:2 = hlfir.declare %[[PDO_PTR_ALLOCA]]
! CHECK: %[[PDO_D_ALLOCA:.*]] = fir.alloca !fir.type<_QMtypesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFtarget_parallel_doEd"}
! CHECK: %[[PDO_D_DECL:.*]]:2 = hlfir.declare %[[PDO_D_ALLOCA]]
! CHECK: %[[PDO_PTR_ARR_ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFtarget_parallel_doEptr_arr"}
! CHECK: %[[PDO_PTR_ARR_DECL:.*]]:2 = hlfir.declare %[[PDO_PTR_ARR_ALLOCA]]
! CHECK: omp.parallel {
! CHECK: fir.store %[[PDO_ASSUMED_DECL]]#0 to %[[PDO_ASSUMED_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: fir.store %[[PDO_AUTO_ARR_DECL]]#0 to %[[PDO_AUTO_ARR_COPY:.*]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK: omp.wsloop private(
! CHECK-SAME: @_QFtarget_parallel_doEs_private_i32 %[[PDO_S_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEarr_private_10xi32 %[[PDO_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEalloc_private_box_heap_i32 %[[PDO_ALLOC_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEptr_private_box_ptr_i32 %[[PDO_PTR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEd_private_rec__QMtypesTdt %[[PDO_D_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEda_private_rec__QMtypesTdt_alloc %[[PDO_DA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEassumed_private_box_Uxi32 %[[PDO_ASSUMED_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEalloc_arr_private_box_heap_Uxi32 %[[PDO_ALLOC_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEptr_arr_private_box_ptr_Uxi32 %[[PDO_PTR_ARR_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEdaa_private_rec__QMtypesTdt_alloc_arr %[[PDO_DAA_DECL]]#0 -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEauto_arr_private_box_Uxi32 %[[PDO_AUTO_ARR_COPY]] -> %arg{{[0-9]+}},
! CHECK-SAME: @_QFtarget_parallel_doEi_private_i32 %[[PDO_I_DECL]]#0 -> %arg{{[0-9]+}} : !fir.ref<i32>, !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.type<_QMtypesTdt{a:i32,b:f32}>>, !fir.ref<!fir.type<_QMtypesTdt_alloc{a:!fir.box<!fir.heap<i32>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.type<_QMtypesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>, !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.ref<i32>) {
