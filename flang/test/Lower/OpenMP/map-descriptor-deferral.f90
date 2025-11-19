!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! This test checks that the descriptor deferral behaviour of the
! MapInfoFinalization pass is preserved. Descriptor deferral is the
! act of removing the mapping of the descriptor in certain cases when
! a descriptor carrying type is mapped. This only applies in certain
! cases and to assumed shape and size dummy arguments that are not
! allocatable or pointers.

subroutine assume_map_target_enter_exit(assumed_arr)
    integer :: assumed_arr(:)
    !$omp target enter data map(to: assumed_arr)
    !$omp target
        assumed_arr(1) = 10
    !$omp end target
    !$omp target exit data map(from: assumed_arr)
end subroutine

!CHECK-LABEL:   func.func @_QPassume_map_target_enter_exit(
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[LOAD_BOX:.*]] = fir.load %[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_ADDR:.*]] = omp.map.info var_ptr(%[[LOAD_BOX]] : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(to) capture(ByRef) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "assumed_arr"}
!CHECK:    omp.target_enter_data map_entries(%[[MAP_ADDR]] : !fir.ref<!fir.array<?xi32>>)
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_ADDR:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[MAP_BOX:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(implicit, to) capture(ByRef) members(%{{.*}} : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "assumed_arr"}
!CHECK:    omp.target map_entries(%[[MAP_BOX]] -> %{{.*}}, %[[MAP_ADDR]] -> %{{.*}} : !fir.ref<!fir.array<?xi32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[LOAD_BOX:.*]] = fir.load %[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_ADDR:.*]] = omp.map.info var_ptr(%[[LOAD_BOX]] : !fir.ref<!fir.array<?xi32>>, i32) map_clauses(from) capture(ByRef) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "assumed_arr"}
!CHECK:    omp.target_exit_data map_entries(%[[MAP_ADDR]] : !fir.ref<!fir.array<?xi32>>)

subroutine assume_alloca_map_target_enter_exit(assumed_arr)
    integer, allocatable :: assumed_arr(:)
    !$omp target enter data map(to: assumed_arr)
    !$omp target
        assumed_arr(1) = 10
    !$omp end target
    !$omp target exit data map(from: assumed_arr)
end subroutine

!CHECK-LABEL:   func.func @_QPassume_alloca_map_target_enter_exit(
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target_enter_data map_entries(%[[DESC_MAP]], %[[BOX_ADDR_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target map_entries(%[[DESC_MAP]] -> %[[VAL_28:.*]], %[[BOX_ADDR_MAP]] -> %[[VAL_29:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(from) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(from) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target_exit_data map_entries(%[[DESC_MAP]], %[[BOX_ADDR_MAP]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)

subroutine assume_pointer_map_target_enter_exit(assumed_arr)
    integer, pointer :: assumed_arr(:)
    !$omp target enter data map(to: assumed_arr)
    !$omp target
        assumed_arr(1) = 10
    !$omp end target
    !$omp target exit data map(from: assumed_arr)
end subroutine

!CHECK-LABEL:   func.func @_QPassume_pointer_map_target_enter_exit(
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, i32) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target_enter_data map_entries(%[[DESC_MAP]], %[[BOX_ADDR_MAP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target map_entries(%[[DESC_MAP]] -> %[[VAL_28:.*]], %[[BOX_ADDR_MAP]] -> %[[VAL_29:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[BOX_ADDR_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, i32) map_clauses(from) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[DESC_MAP:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(from) capture(ByRef) members(%[[BOX_ADDR_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "assumed_arr"}
!CHECK:    omp.target_exit_data map_entries(%[[DESC_MAP]], %[[BOX_ADDR_MAP]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>)

subroutine assume_map_target_data(assumed_arr)
    integer :: assumed_arr(:)
    !$omp target data map(to: assumed_arr)
    !$omp target
        assumed_arr(1) = 10
    !$omp end target
    !$omp end target data
end subroutine

!CHECK-LABEL:   func.func @_QPassume_map_target_data(
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_ADDR:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, i32) map_clauses(to) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[MAP_BOX:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(to) capture(ByRef) members(%[[MAP_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "assumed_arr"}
!CHECK:    omp.target_data map_entries(%[[MAP_BOX]], %[[MAP_ADDR]] : !fir.ref<!fir.array<?xi32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %{{.*}} base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_ADDR:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, i32) map_clauses(implicit, tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!CHECK:    %[[MAP_BOX:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(implicit, to) capture(ByRef) members(%[[MAP_ADDR]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "assumed_arr"}
!CHECK:    omp.target map_entries(%[[MAP_BOX]] -> %{{.*}}, %[[MAP_ADDR]] -> %{{.*}} : !fir.ref<!fir.array<?xi32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
