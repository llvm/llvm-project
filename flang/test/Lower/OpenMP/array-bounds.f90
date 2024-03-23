!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes HOST


!HOST-LABEL:  func.func @_QPread_write_section() {
!HOST:  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFread_write_sectionEi"}
!HOST:  %[[READ:.*]] = fir.address_of(@_QFread_write_sectionEsp_read) : !fir.ref<!fir.array<10xi32>>
!HOST:  %[[C10:.*]] = arith.constant 10 : index
!HOST:  %[[READ_SHAPE:.*]] = fir.shape %[[C10]] : (index) -> !fir.shape<1>
!HOST:  %[[READ_DECL:.*]]:2 = hlfir.declare %[[READ]](%[[READ_SHAPE]]) {uniq_name = "_QFread_write_sectionEsp_read"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!HOST:  %[[WRITE:.*]] = fir.address_of(@_QFread_write_sectionEsp_write) : !fir.ref<!fir.array<10xi32>>
!HOST:  %[[C10_0:.*]] = arith.constant 10 : index
!HOST:  %[[WRITE_SHAPE:.*]] = fir.shape %[[C10_0]] : (index) -> !fir.shape<1>
!HOST:  %[[WRITE_DECL:.*]]:2 = hlfir.declare %[[WRITE]](%[[WRITE_SHAPE]]) {uniq_name = "_QFread_write_sectionEsp_write"} : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>)
!HOST:  %[[C1:.*]] = arith.constant 1 : index
!HOST:  %[[C2:.*]] = arith.constant 1 : index
!HOST:  %[[C3:.*]] = arith.constant 4 : index
!HOST:  %[[BOUNDS0:.*]] = omp.map.bounds   lower_bound(%[[C2]] : index) upper_bound(%[[C3]] : index) extent(%[[C10]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
!HOST:  %[[MAP0:.*]] = omp.map.info var_ptr(%[[READ_DECL]]#0 : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!HOST:  %[[C4:.*]] = arith.constant 1 : index
!HOST:  %[[C5:.*]] = arith.constant 1 : index
!HOST:  %[[C6:.*]] = arith.constant 4 : index
!HOST:  %[[BOUNDS1:.*]] = omp.map.bounds   lower_bound(%[[C5]] : index) upper_bound(%[[C6]] : index) extent(%[[C10_0]] : index) stride(%[[C4]] : index) start_idx(%[[C4]] : index)
!HOST:  %[[MAP1:.*]] = omp.map.info var_ptr(%[[WRITE_DECL]]#0 : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!HOST:  omp.target map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP1]] -> %{{.*}}, {{.*}} -> {{.*}} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, !fir.ref<i32>) {

subroutine read_write_section()
    integer :: sp_read(10) = (/1,2,3,4,5,6,7,8,9,10/)
    integer :: sp_write(10) = (/0,0,0,0,0,0,0,0,0,0/)

!$omp target map(tofrom:sp_read(2:5)) map(tofrom:sp_write(2:5))
    do i = 2, 5
        sp_write(i) = sp_read(i)
    end do
!$omp end target
end subroutine read_write_section


module assumed_array_routines
    contains

!HOST-LABEL: func.func @_QMassumed_array_routinesPassumed_shape_array(
!HOST-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
!HOST: %[[INTERMEDIATE_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!HOST: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMassumed_array_routinesFassumed_shape_arrayEarr_read_write"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
!HOST: %[[C0:.*]] = arith.constant 1 : index
!HOST: %[[C1:.*]] = arith.constant 0 : index
!HOST: %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0_DECL]]#0, %[[C1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!HOST: %[[C3:.*]] = arith.constant 1 : index
!HOST: %[[C4:.*]] = arith.constant 4 : index
!HOST: %[[C0_1:.*]] = arith.constant 0 : index
!HOST: %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0_DECL]]#1, %[[C0_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!HOST: %[[BOUNDS:.*]] = omp.map.bounds   lower_bound(%[[C3]] : index) upper_bound(%[[C4]] : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS0]]#2 : index) start_idx(%[[C0]] : index) {stride_in_bytes = true}
!HOST: %[[VAR_PTR_PTR:.*]] = fir.box_offset %0 base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!HOST: %[[MAP_INFO_MEMBER:.*]] = omp.map.info var_ptr(%[[INTERMEDIATE_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.array<?xi32>) var_ptr_ptr(%[[VAR_PTR_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[INTERMEDIATE_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_INFO_MEMBER]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!HOST: omp.target   map_entries(%[[MAP_INFO_MEMBER]] -> %{{.*}}, %[[MAP]] -> %{{.*}}, {{.*}} -> {{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
    subroutine assumed_shape_array(arr_read_write)
            integer, intent(inout) :: arr_read_write(:)

        !$omp target map(tofrom:arr_read_write(2:5))
            do i = 2, 5
                arr_read_write(i) = i
            end do
        !$omp end target
    end subroutine assumed_shape_array


!HOST-LABEL: func.func @_QMassumed_array_routinesPassumed_size_array(
!HOST-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
!HOST: %[[INTERMEDIATE_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!HOST: %[[ARG0_SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
!HOST: %[[ARG0_DECL:.*]]:2 = hlfir.declare %[[ARG0]](%[[ARG0_SHAPE]]) {fortran_attrs = #fir.var_attrs<intent_inout>, uniq_name = "_QMassumed_array_routinesFassumed_size_arrayEarr_read_write"} : (!fir.ref<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ref<!fir.array<?xi32>>)
!HOST: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_size_arrayEi"}
!HOST: %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0_DECL]]#0, %c0{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!HOST: %[[C4_1:.*]] = arith.subi %c4, %c1{{.*}} : index
!HOST: %[[EXT:.*]] = arith.addi %[[C4_1]], %c1{{.*}} : index
!HOST: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%c1{{.*}} : index) upper_bound(%c4{{.*}} : index) extent(%[[EXT]] : index) stride(%[[DIMS0]]#2 : index) start_idx(%c1{{.*}} : index) {stride_in_bytes = true}
!HOST: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[INTERMEDIATE_ALLOCA]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!HOST: %[[MAP_INFO_MEMBER:.*]] = omp.map.info var_ptr(%[[INTERMEDIATE_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.array<?xi32>) var_ptr_ptr(%[[VAR_PTR_PTR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!HOST: %[[MAP:.*]] = omp.map.info var_ptr(%[[INTERMEDIATE_ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_INFO_MEMBER]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!HOST: omp.target map_entries(%[[MAP_INFO_MEMBER]] -> %{{.*}}, %[[MAP]] -> %{{.*}}, {{.*}} -> {{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
    subroutine assumed_size_array(arr_read_write)
        integer, intent(inout) :: arr_read_write(*)

    !$omp target map(tofrom:arr_read_write(2:5))
        do i = 2, 5
            arr_read_write(i) = i
        end do
    !$omp end target
    end subroutine assumed_size_array
end module assumed_array_routines

!HOST-LABEL:func.func @_QPcall_assumed_shape_and_size_array() {
!HOST: %[[C20:.*]] = arith.constant 20 : index
!HOST: %[[READ_WRITE:.*]] = fir.alloca !fir.array<20xi32> {bindc_name = "arr_read_write", uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"}
!HOST: %[[SHAPE_READ_WRITE:.*]] = fir.shape %[[C20]] : (index) -> !fir.shape<1>
!HOST: %[[READ_WRITE_DECL:.*]]:2 = hlfir.declare %[[READ_WRITE]](%[[SHAPE_READ_WRITE]]) {uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"} : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<20xi32>>, !fir.ref<!fir.array<20xi32>>)
!HOST: %[[LB:.*]] = arith.constant 1 : index
!HOST: %[[UB:.*]] = arith.constant 10 : index
!HOST: %[[STEP:.*]] = arith.constant 1 : index
!HOST: %[[SHAPE_VAL10:.*]] = arith.constant 10 : index
!HOST: %[[SHAPE:.*]] = fir.shape %[[SHAPE_VAL10]] : (index) -> !fir.shape<1>
!HOST: %[[READ_WRITE_DESIGNATE:.*]] = hlfir.designate %[[READ_WRITE_DECL]]#0 (%[[LB]]:%[[UB]]:%[[STEP]])  shape %[[SHAPE]] : (!fir.ref<!fir.array<20xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<10xi32>>
!HOST: %[[READ_WRITE_EMBOX:.*]] = fir.embox %[[READ_WRITE_DESIGNATE]](%[[SHAPE]]) : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xi32>>
!HOST: %[[ARG0:.*]] = fir.convert %[[READ_WRITE_EMBOX]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
!HOST: fir.call @_QMassumed_array_routinesPassumed_shape_array(%[[ARG0]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
!HOST: %[[LB:.*]] = arith.constant 10 : index
!HOST: %[[UB:.*]] = arith.constant 20 : index
!HOST: %[[STEP:.*]] = arith.constant 1 : index
!HOST: %[[SHAPE_VAL11:.*]] = arith.constant 11 : index
!HOST: %[[SHAPE:.*]] = fir.shape %[[SHAPE_VAL11]] : (index) -> !fir.shape<1>
!HOST: %[[READ_WRITE_EMBOX:.*]] = hlfir.designate %[[READ_WRITE_DECL]]#0 (%[[LB]]:%[[UB]]:%[[STEP]])  shape %[[SHAPE]] : (!fir.ref<!fir.array<20xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<11xi32>>
!HOST: %[[ARG1:.*]] = fir.convert %[[READ_WRITE_EMBOX]] : (!fir.ref<!fir.array<11xi32>>) -> !fir.ref<!fir.array<?xi32>>
!HOST: fir.call @_QMassumed_array_routinesPassumed_size_array(%[[ARG1]]) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
!HOST: return
!HOST:}

subroutine call_assumed_shape_and_size_array
    use assumed_array_routines
    integer :: arr_read_write(20)
    call assumed_shape_array(arr_read_write(1:10))
    call assumed_size_array(arr_read_write(10:20))
end subroutine call_assumed_shape_and_size_array
