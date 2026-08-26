! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module nested_private_types
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

subroutine nested_target_teams(n, assumed)
  use nested_private_types
  integer :: n, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target
    !$omp teams private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
      s = 1
    !$omp end teams
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_teams
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: omp.map.info {{.*}} name("alloc")
! CHECK: omp.map.info {{.*}} name("da")
! CHECK: omp.map.info {{.*}} name("assumed")
! CHECK: omp.map.info {{.*}} name("alloc_arr")
! CHECK: omp.map.info {{.*}} name("daa")
! CHECK: omp.map.info {{.*}} name("auto_arr")
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: fir.alloca i32 {bindc_name = "s", pinned, uniq_name = "_QFnested_target_teamsEs"}
! CHECK: fir.alloca !fir.array<10xi32> {bindc_name = "arr", pinned, uniq_name = "_QFnested_target_teamsEarr"}
! CHECK: fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "alloc", pinned, uniq_name = "_QFnested_target_teamsEalloc"}
! CHECK: fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "ptr", pinned, uniq_name = "_QFnested_target_teamsEptr"}
! CHECK: fir.alloca !fir.type<_QMnested_private_typesTdt{a:i32,b:f32}> {bindc_name = "d", pinned, uniq_name = "_QFnested_target_teamsEd"}
! CHECK: fir.alloca !fir.type<_QMnested_private_typesTdt_alloc{a:!fir.box<!fir.heap<i32>>}> {bindc_name = "da", pinned, uniq_name = "_QFnested_target_teamsEda"}
! CHECK: fir.alloca !fir.array<?xi32>, %{{.*}} {bindc_name = "assumed", pinned, uniq_name = "_QFnested_target_teamsEassumed"}
! CHECK: fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "alloc_arr", pinned, uniq_name = "_QFnested_target_teamsEalloc_arr"}
! CHECK: fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "ptr_arr", pinned, uniq_name = "_QFnested_target_teamsEptr_arr"}
! CHECK: fir.alloca !fir.type<_QMnested_private_typesTdt_alloc_arr{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {bindc_name = "daa", pinned, uniq_name = "_QFnested_target_teamsEdaa"}
! CHECK: fir.alloca !fir.array<?xi32>, %{{.*}} {bindc_name = "auto_arr", pinned, uniq_name = "_QFnested_target_teamsEauto_arr"}

subroutine nested_target_parallel(n, assumed)
  use nested_private_types
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

! CHECK-LABEL: func.func @_QPnested_target_parallel
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: omp.map.info {{.*}} name("alloc")
! CHECK: omp.map.info {{.*}} name("da")
! CHECK: omp.map.info {{.*}} name("assumed")
! CHECK: omp.map.info {{.*}} name("alloc_arr")
! CHECK: omp.map.info {{.*}} name("daa")
! CHECK: omp.map.info {{.*}} name("auto_arr")
! CHECK: omp.target
! CHECK: omp.parallel private(
! CHECK-SAME: @_QFnested_target_parallelEs_private_i32
! CHECK-SAME: @_QFnested_target_parallelEarr_private_10xi32
! CHECK-SAME: @_QFnested_target_parallelEalloc_private_box_heap_i32
! CHECK-SAME: @_QFnested_target_parallelEptr_private_box_ptr_i32
! CHECK-SAME: @_QFnested_target_parallelEd_private_rec__QMnested_private_typesTdt
! CHECK-SAME: @_QFnested_target_parallelEda_private_rec__QMnested_private_typesTdt_alloc
! CHECK-SAME: @_QFnested_target_parallelEassumed_private_box_Uxi32
! CHECK-SAME: @_QFnested_target_parallelEalloc_arr_private_box_heap_Uxi32
! CHECK-SAME: @_QFnested_target_parallelEptr_arr_private_box_ptr_Uxi32
! CHECK-SAME: @_QFnested_target_parallelEdaa_private_rec__QMnested_private_typesTdt_alloc_arr
! CHECK-SAME: @_QFnested_target_parallelEauto_arr_private_box_Uxi32

subroutine nested_target_teams_distribute(n, assumed)
  use nested_private_types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target
    !$omp teams distribute private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
    do i = 1, n
      s = i
    end do
    !$omp end teams distribute
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_teams_distribute
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: omp.map.info {{.*}} name("alloc")
! CHECK: omp.map.info {{.*}} name("da")
! CHECK: omp.map.info {{.*}} name("assumed")
! CHECK: omp.map.info {{.*}} name("alloc_arr")
! CHECK: omp.map.info {{.*}} name("daa")
! CHECK: omp.map.info {{.*}} name("auto_arr")
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.distribute private(
! CHECK-SAME: @_QFnested_target_teams_distributeEs_private_i32
! CHECK-SAME: @_QFnested_target_teams_distributeEarr_private_10xi32
! CHECK-SAME: @_QFnested_target_teams_distributeEalloc_private_box_heap_i32
! CHECK-SAME: @_QFnested_target_teams_distributeEptr_private_box_ptr_i32
! CHECK-SAME: @_QFnested_target_teams_distributeEd_private_rec__QMnested_private_typesTdt
! CHECK-SAME: @_QFnested_target_teams_distributeEda_private_rec__QMnested_private_typesTdt_alloc
! CHECK-SAME: @_QFnested_target_teams_distributeEassumed_private_box_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distributeEalloc_arr_private_box_heap_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distributeEptr_arr_private_box_ptr_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distributeEdaa_private_rec__QMnested_private_typesTdt_alloc_arr
! CHECK-SAME: @_QFnested_target_teams_distributeEauto_arr_private_box_Uxi32

subroutine nested_target_teams_distribute_simd(n, assumed)
  use nested_private_types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target
    !$omp teams distribute simd private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
    do i = 1, n
      s = i
    end do
    !$omp end teams distribute simd
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_teams_distribute_simd
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: omp.map.info {{.*}} name("alloc")
! CHECK: omp.map.info {{.*}} name("da")
! CHECK: omp.map.info {{.*}} name("assumed")
! CHECK: omp.map.info {{.*}} name("alloc_arr")
! CHECK: omp.map.info {{.*}} name("daa")
! CHECK: omp.map.info {{.*}} name("auto_arr")
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.distribute
! CHECK: omp.simd private(
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEs_private_i32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEarr_private_10xi32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEalloc_private_box_heap_i32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEptr_private_box_ptr_i32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEd_private_rec__QMnested_private_typesTdt
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEda_private_rec__QMnested_private_typesTdt_alloc
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEassumed_private_box_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEalloc_arr_private_box_heap_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEptr_arr_private_box_ptr_Uxi32
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEdaa_private_rec__QMnested_private_typesTdt_alloc_arr
! CHECK-SAME: @_QFnested_target_teams_distribute_simdEauto_arr_private_box_Uxi32

subroutine nested_target_parallel_do(n, assumed)
  use nested_private_types
  integer :: n, i, s, arr(10), assumed(:)
  integer, allocatable :: alloc, alloc_arr(:)
  integer, pointer :: ptr, ptr_arr(:)
  type(dt) :: d
  type(dt_alloc) :: da
  type(dt_alloc_arr) :: daa
  integer :: auto_arr(n)
  !$omp target
    !$omp parallel do private(s, arr, alloc, ptr, d, da, assumed, alloc_arr, ptr_arr, daa, auto_arr)
    do i = 1, n
      s = i
    end do
    !$omp end parallel do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_parallel_do
! CHECK-NOT: omp.map.info {{.*}} name("s")
! CHECK-NOT: omp.map.info {{.*}} name("arr")
! CHECK-NOT: omp.map.info {{.*}} name("ptr")
! CHECK-NOT: omp.map.info {{.*}} name("d")
! CHECK-NOT: omp.map.info {{.*}} name("ptr_arr")
! CHECK: omp.map.info {{.*}} name("alloc")
! CHECK: omp.map.info {{.*}} name("da")
! CHECK: omp.map.info {{.*}} name("assumed")
! CHECK: omp.map.info {{.*}} name("alloc_arr")
! CHECK: omp.map.info {{.*}} name("daa")
! CHECK: omp.map.info {{.*}} name("auto_arr")
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK: omp.wsloop private(
! CHECK-SAME: @_QFnested_target_parallel_doEs_private_i32
! CHECK-SAME: @_QFnested_target_parallel_doEarr_private_10xi32
! CHECK-SAME: @_QFnested_target_parallel_doEalloc_private_box_heap_i32
! CHECK-SAME: @_QFnested_target_parallel_doEptr_private_box_ptr_i32
! CHECK-SAME: @_QFnested_target_parallel_doEd_private_rec__QMnested_private_typesTdt
! CHECK-SAME: @_QFnested_target_parallel_doEda_private_rec__QMnested_private_typesTdt_alloc
! CHECK-SAME: @_QFnested_target_parallel_doEassumed_private_box_Uxi32
! CHECK-SAME: @_QFnested_target_parallel_doEalloc_arr_private_box_heap_Uxi32
! CHECK-SAME: @_QFnested_target_parallel_doEptr_arr_private_box_ptr_Uxi32
! CHECK-SAME: @_QFnested_target_parallel_doEdaa_private_rec__QMnested_private_typesTdt_alloc_arr
! CHECK-SAME: @_QFnested_target_parallel_doEauto_arr_private_box_Uxi32

subroutine nested_target_parallel_intervening_code()
  integer :: guard, k
  !$omp target
    guard = 0
    !$omp parallel private(k)
      k = 10
    !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_parallel_intervening_code
! CHECK: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: omp.parallel private({{.*}}@_QFnested_target_parallel_intervening_codeEk_private_i32

subroutine nested_target_parallel_do_intervening_code()
  integer :: guard, i, k
  !$omp target
    guard = 0
    !$omp parallel do private(k)
    do i = 1, 10
      k = i
    end do
    !$omp end parallel do
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_parallel_do_intervening_code
! CHECK: omp.map.info {{.*}} name("k")
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK: omp.wsloop private({{.*}}@_QFnested_target_parallel_do_intervening_codeEk_private_i32

subroutine nested_target_parallel_both_private()
  integer :: k
  !$omp target private(k)
    !$omp parallel private(k)
      k = 10
    !$omp end parallel
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_parallel_both_private
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.target kernel_type(generic) private(@_QFnested_target_parallel_both_privateEk_private_i32 %{{.*}}#0 -> %[[TP_K_ARG:arg[0-9]+]] : !fir.ref<i32>) {
! CHECK: %[[TP_K_DECL:.*]]:2 = hlfir.declare %[[TP_K_ARG]] {uniq_name = "_QFnested_target_parallel_both_privateEk"}
! CHECK: omp.parallel private(@_QFnested_target_parallel_both_privateEk_private_i32 %[[TP_K_DECL]]#0 -> %[[PAR_K_ARG:arg[0-9]+]] : !fir.ref<i32>) {
! CHECK: hlfir.declare %[[PAR_K_ARG]] {uniq_name = "_QFnested_target_parallel_both_privateEk"}

subroutine nested_target_teams_distribute_both_private()
  integer :: i, k
  !$omp target private(k)
    !$omp teams distribute private(k)
    do i = 1, 10
      k = i
    end do
    !$omp end teams distribute
  !$omp end target
end subroutine

! CHECK-LABEL: func.func @_QPnested_target_teams_distribute_both_private
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.map.info {{.*}} name("i")
! CHECK-NOT: omp.map.info {{.*}} name("k")
! CHECK: omp.target {{.*}}private(@_QFnested_target_teams_distribute_both_privateEk_private_i32 %{{.*}}#0 -> %[[TTD_K_ARG:arg[0-9]+]]
! CHECK: %[[TTD_K_DECL:.*]]:2 = hlfir.declare %[[TTD_K_ARG]] {uniq_name = "_QFnested_target_teams_distribute_both_privateEk"}
! CHECK: omp.teams
! CHECK: omp.distribute private({{.*}}@_QFnested_target_teams_distribute_both_privateEk_private_i32 %[[TTD_K_DECL]]#0 -> %[[DIST_K_ARG:arg[0-9]+]]
! CHECK: hlfir.declare %[[DIST_K_ARG]] {uniq_name = "_QFnested_target_teams_distribute_both_privateEk"}
