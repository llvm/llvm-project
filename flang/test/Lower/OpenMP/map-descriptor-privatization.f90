!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! This test checks the descriptor privatization for assumed arrays verifying the
! maps have the appropriate map types applied to undergo attach map privatization.

subroutine assumed_shape_array_priv(arr_read_write)
    integer, intent(inout) :: arr_read_write(:)
    !$omp target map(tofrom: arr_read_write)
        arr_read_write(1) = 10
    !$omp end target
end subroutine

!CHECK-LABEL:   func.func @_QPassumed_shape_array_priv(
!CHECK:    %[[DESC_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!CHECK:    %[[BOX_ADDR:.*]] = fir.box_offset %[[DESC_ALLOCA]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_MEMBER:.*]] = omp.map.info var_ptr(%[[DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) bounds(%{{.*}}) name("") -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK:    %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%[[DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(target_param, private, attach) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) members(%[[MAP_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) name("arr_read_write") -> !fir.ref<!fir.array<?xi32>>
!CHECK:    %[[MAP_ATTACH:.*]] = omp.map.info var_ptr(%[[DESC_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(attach, ref_ptr, ref_ptee) capture(ByRef) var_ptr_ptr(%[[BOX_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, i32) bounds(%{{.*}}) name("arr_read_write") -> !fir.ref<!fir.array<?xi32>>
!CHECK:    omp.target kernel_type(generic) map_entries(%[[MAP_PARENT]] -> %{{.*}}, %[[MAP_ATTACH]] -> %{{.*}}, %[[MAP_MEMBER]] -> %{{.*}} : !fir.ref<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
