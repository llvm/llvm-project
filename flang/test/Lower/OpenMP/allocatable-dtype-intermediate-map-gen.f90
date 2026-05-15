!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine target_map_to()
    type :: dtype
        integer, allocatable :: scalar
    end type dtype

    type(dtype), allocatable :: derived

    allocate(derived%scalar)

!CHECK: %[[SCALAR_DATA:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(to) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[SCALAR_DESC:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(always, to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "derived%scalar"}
!CHECK: %[[SCALAR_DESC_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "derived%scalar"}

!CHECK: %[[DTYPE_DATA:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>>, !fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>) map_clauses(storage) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>, !fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>) -> !fir.llvm_ptr<!fir.ref<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>> {name = ""}
!CHECK: %[[DTYPE_DESC:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>>, !fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>) map_clauses(always, to) capture(ByRef) members({{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>> {name = "derived"}
!CHECK: %[[DTYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>>, !fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>, !fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>> {name = "derived"}

!CHECK: omp.target map_entries(%[[DTYPE_DESC]] -> %{{.*}}, %[[SCALAR_DESC]] -> %{{.*}}, %[[SCALAR_DESC_ATTACH]] -> %{{.*}}, %[[DTYPE_DESC_ATTACH]] -> %{{.*}}, %[[DTYPE_DATA]] -> %{{.*}}, %[[SCALAR_DATA]] -> %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>>, !fir.llvm_ptr<!fir.ref<!fir.type<_QFtarget_map_toTdtype{scalar:!fir.box<!fir.heap<i32>>}>>>, !fir.llvm_ptr<!fir.ref<i32>>) {

!$omp target map(to:derived%scalar)
!$omp end target

end subroutine target_map_to

subroutine update_map_to()
    type :: dtype
        integer, allocatable :: scalar
    end type dtype

    type(dtype), allocatable :: derived

    allocate(derived%scalar)

!CHECK: %[[SCALAR_DATA:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(to) capture(ByRef) var_ptr_ptr({{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
!CHECK: %[[SCALAR_DESC:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(to) capture(ByRef) members{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.heap<i32> {name = "derived%scalar"}
!CHECK: %[[SCALAR_DESC_ATTACH:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%{{.*}} : !fir.llvm_ptr<!fir.ref<i32>>, i32) -> !fir.heap<i32> {name = "derived%scalar"}

!CHECK: omp.target_update map_entries(%[[SCALAR_DESC]], %[[SCALAR_DESC_ATTACH]], %[[SCALAR_DATA]] : !fir.heap<i32>, !fir.heap<i32>, !fir.llvm_ptr<!fir.ref<i32>>)

!$omp target update to(derived%scalar)

end subroutine update_map_to
