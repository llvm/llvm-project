!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 %s -o - | FileCheck %s

subroutine defaultmap_allocatable_present()
    implicit none
    integer, dimension(:), allocatable :: arr

! CHECK: %[[MAP_1:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(implicit, present, exit_release_or_enter_alloc) capture(ByRef) var_ptr_ptr({{.*}}) bounds({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
! CHECK: %[[MAP_2:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members({{.*}}) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "arr"}
!$omp target defaultmap(present: allocatable)
    arr(1) = 10
!$omp end target

    return
end subroutine

subroutine defaultmap_scalar_tofrom()
    implicit none
    integer :: scalar_int

! CHECK: %[[MAP:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(implicit, tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_int"}
   !$omp target defaultmap(tofrom: scalar)
        scalar_int = 20
   !$omp end target

    return
end subroutine

subroutine defaultmap_all_default()
    implicit none
    integer, dimension(:), allocatable :: arr
    integer :: aggregate(16)
    integer :: scalar_int

! CHECK: %[[MAP_1:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "scalar_int"}
! CHECK: %[[MAP_2:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr({{.*}}) bounds({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
! CHECK: %[[MAP_3:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members({{.*}}) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "arr"}
! CHECK: %[[MAP_4:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(implicit, tofrom) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "aggregate"}

   !$omp target defaultmap(default: all)
        scalar_int = 20
        arr(1) = scalar_int + aggregate(1)
   !$omp end target

    return
end subroutine

subroutine defaultmap_pointer_to()
    implicit none
    integer, dimension(:), pointer :: arr_ptr(:)
    integer :: scalar_int

! CHECK: %[[MAP_1:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, i32) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr({{.*}}) bounds({{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
! CHECK: %[[MAP_2:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members({{.*}}) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "arr_ptr"}
! CHECK: %[[MAP_3:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "scalar_int"}
    !$omp target defaultmap(to: pointer)
        arr_ptr(1) = scalar_int + 20
    !$omp end target

    return
end subroutine

subroutine defaultmap_scalar_from()
    implicit none
    integer :: scalar_test

! CHECK:%[[MAP:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(implicit, from) capture(ByRef) -> !fir.ref<i32> {name = "scalar_test"}
    !$omp target defaultmap(from: scalar)
        scalar_test = 20
    !$omp end target

    return
end subroutine

subroutine defaultmap_aggregate_to()
    implicit none
    integer :: aggregate_arr(16)
    integer :: scalar_test

! CHECK: %[[MAP_1:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "scalar_test"}
! CHECK: %[[MAP_2:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.array<16xi32>>, !fir.array<16xi32>) map_clauses(implicit, to) capture(ByRef) bounds({{.*}}) -> !fir.ref<!fir.array<16xi32>> {name = "aggregate_arr"}
    !$omp target map(tofrom: scalar_test) defaultmap(to: aggregate)
        aggregate_arr(1) = 1
        scalar_test = 1
    !$omp end target

    return
end subroutine

subroutine defaultmap_dtype_aggregate_to()
    implicit none
    type :: dtype
        integer(4) :: array_i(10)
        integer(4) :: k
    end type dtype

    type(dtype) :: aggregate_type

! CHECK: %[[MAP:.*]] = omp.map.info var_ptr({{.*}} : !fir.ref<!fir.type<_QFdefaultmap_dtype_aggregate_toTdtype{array_i:!fir.array<10xi32>,k:i32}>>, !fir.type<_QFdefaultmap_dtype_aggregate_toTdtype{array_i:!fir.array<10xi32>,k:i32}>) map_clauses(implicit, to) capture(ByRef) -> !fir.ref<!fir.type<_QFdefaultmap_dtype_aggregate_toTdtype{array_i:!fir.array<10xi32>,k:i32}>> {name = "aggregate_type"}
    !$omp target defaultmap(to: aggregate)
        aggregate_type%k = 40
        aggregate_type%array_i(1) = 50
    !$omp end target

    return
end subroutine
