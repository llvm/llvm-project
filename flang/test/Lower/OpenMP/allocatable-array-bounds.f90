!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes HOST

!HOST-LABEL:  func.func @_QPread_write_section() {

!HOST: %[[ALLOCA_1:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "sp_read", uniq_name = "_QFread_write_sectionEsp_read"}
!HOST: %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_1]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFread_write_sectionEsp_read"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)

!HOST: %[[ALLOCA_2:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "sp_write", uniq_name = "_QFread_write_sectionEsp_write"}
!HOST: %[[DECLARE_2:.*]]:2 = hlfir.declare %[[ALLOCA_2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFread_write_sectionEsp_write"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)

!HOST: %[[LOAD_1:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[LOAD_2:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_1:.*]] = arith.constant 0 : index
!HOST: %[[BOX_1:.*]]:3 = fir.box_dims %[[LOAD_2]], %[[CONSTANT_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_2:.*]] = arith.constant 0 : index
!HOST: %[[BOX_2:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_3:.*]] = arith.constant 2 : index
!HOST: %[[LB_1:.*]] = arith.subi %[[CONSTANT_3]], %[[BOX_1]]#0 : index
!HOST: %[[CONSTANT_4:.*]] = arith.constant 5 : index
!HOST: %[[UB_1:.*]] = arith.subi %[[CONSTANT_4]], %[[BOX_1]]#0 : index
!HOST: %[[LOAD_3:.*]] = fir.load %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_3:.*]] = arith.constant 0 : index
!HOST: %[[BOX_3:.*]]:3 = fir.box_dims %[[LOAD_3]], %[[CONSTANT_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[BOUNDS_1:.*]] = omp.map.bounds lower_bound(%[[LB_1]] : index) upper_bound(%[[UB_1]] : index) extent(%[[BOX_3]]#1 : index) stride(%[[BOX_2]]#2 : index) start_idx(%[[BOX_1]]#0 : index) {stride_in_bytes = true}
!HOST: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[DECLARE_1]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!HOST: %[[MAP_INFO_MEMBER:.*]] = omp.map.info var_ptr(%[[DECLARE_1]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAR_PTR_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS_1]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!HOST: %[[MAP_INFO_1:.*]] = omp.map.info var_ptr(%[[DECLARE_1]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%[[MAP_INFO_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "sp_read(2:5)"}

!HOST: %[[LOAD_3:.*]] = fir.load %[[DECLARE_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[LOAD_4:.*]] = fir.load %[[DECLARE_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_5:.*]] = arith.constant 0 : index
!HOST: %[[BOX_3:.*]]:3 = fir.box_dims %[[LOAD_4]], %[[CONSTANT_5]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_6:.*]] = arith.constant 0 : index
!HOST: %[[BOX_4:.*]]:3 = fir.box_dims %[[LOAD_3]], %[[CONSTANT_6]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_7:.*]] = arith.constant 2 : index
!HOST: %[[LB_2:.*]] = arith.subi %[[CONSTANT_7]], %[[BOX_3]]#0 : index
!HOST: %[[CONSTANT_8:.*]] = arith.constant 5 : index
!HOST: %[[UB_2:.*]] = arith.subi %[[CONSTANT_8]], %[[BOX_3]]#0 : index
!HOST: %[[LOAD_5:.*]] = fir.load %[[DECLARE_2]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_5:.*]] = arith.constant 0 : index
!HOST: %[[BOX_5:.*]]:3 = fir.box_dims %[[LOAD_5]], %[[CONSTANT_5]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[BOUNDS_2:.*]] = omp.map.bounds lower_bound(%[[LB_2]] : index) upper_bound(%[[UB_2]] : index) extent(%[[BOX_5]]#1 : index) stride(%[[BOX_4]]#2 : index) start_idx(%[[BOX_3]]#0 : index) {stride_in_bytes = true}
!HOST: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[DECLARE_2]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!HOST: %[[MAP_INFO_MEMBER:.*]] = omp.map.info var_ptr(%[[DECLARE_2]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAR_PTR_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS_2]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!HOST: %[[MAP_INFO_2:.*]] = omp.map.info var_ptr(%[[DECLARE_2]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%[[MAP_INFO_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "sp_write(2:5)"}

subroutine read_write_section()
    integer, allocatable :: sp_read(:)
    integer, allocatable :: sp_write(:)
    allocate(sp_read(10)) 
    allocate(sp_write(10))
    sp_write = (/0,0,0,0,0,0,0,0,0,0/)
    sp_read = (/1,2,3,4,5,6,7,8,9,10/)

!$omp target map(tofrom:sp_read(2:5)) map(tofrom:sp_write(2:5))
    do i = 2, 5
        sp_write(i) = sp_read(i)
    end do
!$omp end target
end subroutine read_write_section

module assumed_allocatable_array_routines
    contains

!HOST-LABEL: func.func @_QMassumed_allocatable_array_routinesPassumed_shape_array(

!HOST: %[[DECLARE:.*]]:2 = hlfir.declare %[[ARG:.*]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable, intent_inout>, uniq_name = "_QMassumed_allocatable_array_routinesFassumed_shape_arrayEarr_read_write"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!HOST: %[[LOAD_1:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[LOAD_2:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_1:.*]] = arith.constant 0 : index
!HOST: %[[BOX_1:.*]]:3 = fir.box_dims %[[LOAD_2]], %[[CONSTANT_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_2:.*]] = arith.constant 0 : index
!HOST: %[[BOX_2:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[CONSTANT_3:.*]] = arith.constant 2 : index
!HOST: %[[LB:.*]] = arith.subi %[[CONSTANT_3]], %[[BOX_1]]#0 : index
!HOST: %[[CONSTANT_4:.*]] = arith.constant 5 : index
!HOST: %[[UB:.*]] = arith.subi %[[CONSTANT_4]], %[[BOX_1]]#0 : index
!HOST: %[[LOAD_3:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[CONSTANT_3:.*]] = arith.constant 0 : index
!HOST: %[[BOX_3:.*]]:3 = fir.box_dims %[[LOAD_3]], %[[CONSTANT_3]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
!HOST: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%[[LB]] : index) upper_bound(%[[UB]] : index) extent(%[[BOX_3]]#1 : index) stride(%[[BOX_2]]#2 : index) start_idx(%[[BOX_1]]#0 : index) {stride_in_bytes = true}
!HOST: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!HOST: %[[MAP_INFO_MEMBER:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAR_PTR_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!HOST: %[[MAP_INFO:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) members(%[[MAP_INFO_MEMBER]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "arr_read_write(2:5)"}

subroutine assumed_shape_array(arr_read_write)
    integer, allocatable, intent(inout) :: arr_read_write(:)

!$omp target map(tofrom:arr_read_write(2:5))
    do i = 2, 5
        arr_read_write(i) = i
    end do
!$omp end target
end subroutine assumed_shape_array
end module assumed_allocatable_array_routines

!HOST-LABEL: func.func @_QPcall_assumed_shape_and_size_array() {
!HOST: %[[VAL_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "arr_read_write", uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"}
!HOST: %[[VAL_1:.*]] = fir.zero_bits !fir.heap<!fir.array<?xi32>>
!HOST: %[[VAL_2:.*]] = arith.constant 0 : index
!HOST: %[[VAL_3:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
!HOST: %[[VAL_4:.*]] = fir.embox %[[VAL_1]](%[[VAL_3]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
!HOST: fir.store %[[VAL_4]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = {{.*}}<allocatable>, uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
!HOST: %[[VAL_6:.*]] = arith.constant false
!HOST: %[[VAL_7:.*]] = fir.absent !fir.box<none>
!HOST: %[[VAL_8:.*]] = fir.address_of(@_QQclX406d77a864a01343bfa72d1eaaccf436) : !fir.ref<!fir.char<1,84>>
!HOST: %[[VAL_9:.*]] = arith.constant {{.*}} : i32
!HOST: %[[VAL_10:.*]] = arith.constant 1 : index
!HOST: %[[VAL_11:.*]] = arith.constant {{.*}} : i32
!HOST: %[[VAL_12:.*]] = arith.constant 0 : i32
!HOST: %[[VAL_13:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
!HOST: %[[VAL_14:.*]] = fir.convert %[[VAL_10]] : (index) -> i64
!HOST: %[[VAL_15:.*]] = fir.convert %[[VAL_11]] : (i32) -> i64
!HOST: fir.call @_FortranAAllocatableSetBounds(%[[VAL_13]], %[[VAL_12]], %[[VAL_14]], %[[VAL_15]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
!HOST: %[[VAL_16:.*]] = fir.convert %[[VAL_5]]#1 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.box<none>>
!HOST: %[[VAL_17:.*]] = fir.convert %[[VAL_8]] : (!fir.ref<!fir.char<1,84>>) -> !fir.ref<i8>
!HOST: %[[VAL_18:.*]] = fir.call @_FortranAAllocatableAllocate(%[[VAL_16]], %[[VAL_6]], %[[VAL_7]], %[[VAL_17]], %[[VAL_9]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i1, !fir.box<none>, !fir.ref<i8>, i32) -> i32
!HOST: %[[VAL_19:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!HOST: %[[VAL_20:.*]] = arith.constant {{.*}} : index
!HOST: %[[VAL_21:.*]] = arith.constant {{.*}} : index
!HOST: %[[VAL_22:.*]] = arith.constant 1 : index
!HOST: %[[VAL_23:.*]] = arith.constant {{.*}} : index
!HOST: %[[VAL_24:.*]] = fir.shape %[[VAL_23]] : (index) -> !fir.shape<1>
!HOST: %[[VAL_25:.*]] = hlfir.designate %[[VAL_19]] (%[[VAL_20]]:%[[VAL_21]]:%[[VAL_22]])  shape %[[VAL_24]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<11xi32>>
!HOST: fir.call @_QPassumed_size_array(%[[VAL_25]]) fastmath<contract> : (!fir.ref<!fir.array<11xi32>>) -> ()
subroutine call_assumed_shape_and_size_array
    use assumed_allocatable_array_routines
    integer, allocatable :: arr_read_write(:)
    allocate(arr_read_write(20))
    call assumed_size_array(arr_read_write(10:20))
    deallocate(arr_read_write)
end subroutine call_assumed_shape_and_size_array
