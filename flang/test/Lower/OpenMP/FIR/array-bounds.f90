!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp %s -o - | FileCheck %s --check-prefixes=HOST,ALL
!RUN: %flang_fc1 -emit-fir -flang-deprecated-no-hlfir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefixes=DEVICE,ALL

!ALL-LABEL: func.func @_QPread_write_section(
!ALL:  %[[ITER:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFread_write_sectionEi"}
!ALL:  %[[READ:.*]] = fir.address_of(@_QFread_write_sectionEsp_read) : !fir.ref<!fir.array<10xi32>>
!ALL:  %[[C10:.*]] = arith.constant 10 : index
!ALL:  %[[WRITE:.*]] = fir.address_of(@_QFread_write_sectionEsp_write) : !fir.ref<!fir.array<10xi32>>
!ALL:  %[[C10_0:.*]] = arith.constant 10 : index
!ALL:  %[[C1:.*]] = arith.constant 1 : index
!ALL:  %[[C2:.*]] = arith.constant 1 : index
!ALL:  %[[C3:.*]] = arith.constant 4 : index
!ALL:  %[[BOUNDS0:.*]] = omp.map.bounds   lower_bound(%[[C2]] : index) upper_bound(%[[C3]] : index) extent(%[[C10]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
!ALL:  %[[MAP0:.*]] = omp.map.info var_ptr(%[[READ]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!ALL:  %[[C4:.*]] = arith.constant 1 : index
!ALL:  %[[C5:.*]] = arith.constant 1 : index
!ALL:  %[[C6:.*]] = arith.constant 4 : index
!ALL:  %[[BOUNDS1:.*]] = omp.map.bounds   lower_bound(%[[C5]] : index) upper_bound(%[[C6]] : index) extent(%[[C10_0]] : index) stride(%[[C4]] : index) start_idx(%[[C4]] : index)
!ALL:  %[[MAP1:.*]] = omp.map.info var_ptr(%[[WRITE]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!ALL:  %[[MAP2:.*]] = omp.map.info var_ptr(%[[ITER]] : !fir.ref<i32>, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
!ALL: omp.target map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP1]] -> %{{.*}}, %[[MAP2]] -> %{{.*}} : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>, !fir.ref<i32>) {

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
!ALL-LABEL: func.func @_QMassumed_array_routinesPassumed_shape_array(
!ALL-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"})
!ALL: %[[INTERMEDIATE_ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
!ALL: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_shape_arrayEi"}
!ALL: %[[C0:.*]] = arith.constant 1 : index
!ALL: %[[C1:.*]] = arith.constant 0 : index
!ALL: %[[DIMS0:.*]]:3 = fir.box_dims %arg0, %[[C1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!ALL: %[[C3:.*]] = arith.constant 1 : index
!ALL: %[[C4:.*]] = arith.constant 4 : index
!ALL: %[[C0_1:.*]] = arith.constant 0 : index
!ALL: %[[DIMS1:.*]]:3 = fir.box_dims %arg0, %[[C0_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!ALL: %[[BOUNDS:.*]] = omp.map.bounds   lower_bound(%[[C3]] : index) upper_bound(%[[C4]] : index) extent(%[[DIMS1]]#1 : index) stride(%[[DIMS0]]#2 : index) start_idx(%[[C0]] : index) {stride_in_bytes = true}
!ALL: %[[BOXADDRADDR:.*]] = fir.box_offset %0 base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!ALL: %[[MAP_MEMBER:.*]] = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.array<?xi32>) var_ptr_ptr(%[[BOXADDRADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
!ALL: %[[MAP:.*]] = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!ALL: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<i32>, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
!ALL: omp.target map_entries(%[[MAP_MEMBER]] -> %{{.*}}, %[[MAP]] -> %{{.*}}, %[[MAP2]] -> %{{.*}} : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
    subroutine assumed_shape_array(arr_read_write)
        integer, intent(inout) :: arr_read_write(:)

        !$omp target map(tofrom:arr_read_write(2:5))
            do i = 2, 5
                arr_read_write(i) = i
            end do
        !$omp end target
    end subroutine assumed_shape_array

!ALL-LABEL:   func.func @_QMassumed_array_routinesPassumed_size_array(
!ALL-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"})
!ALL: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_size_arrayEi"}
!ALL: %[[C0:.*]] = arith.constant 1 : index
!ALL: %[[C1:.*]] = arith.constant 1 : index
!ALL: %[[C2:.*]] = arith.constant 4 : index
!ALL: %[[DIFF:.*]] = arith.subi %[[C2]], %[[C1]] : index
!ALL: %[[EXT:.*]] = arith.addi %[[DIFF]], %[[C0]] : index
!ALL: %[[BOUNDS:.*]]  = omp.map.bounds   lower_bound(%[[C1]] : index) upper_bound(%[[C2]] : index) extent(%[[EXT]] : index) stride(%[[C0]] : index) start_idx(%[[C0]] : index)
!ALL: %[[MAP:.*]] = omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!ALL: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<i32>, i32)   map_clauses(implicit, exit_release_or_enter_alloc) capture(ByCopy) -> !fir.ref<i32> {name = "i"}
!ALL: omp.target map_entries(%[[MAP]] -> %{{.*}}, %[[MAP2]] -> %{{.*}} : !fir.ref<!fir.array<?xi32>>, !fir.ref<i32>) {
    subroutine assumed_size_array(arr_read_write)
        integer, intent(inout) :: arr_read_write(*)

    !$omp target map(tofrom:arr_read_write(2:5))
        do i = 2, 5
            arr_read_write(i) = i
        end do
    !$omp end target
    end subroutine assumed_size_array
end module assumed_array_routines

!DEVICE-NOT:func.func @_QPcall_assumed_shape_and_size_array() {

!HOST-LABEL:func.func @_QPcall_assumed_shape_and_size_array() {
!HOST:%{{.*}} = arith.constant 20 : index
!HOST:%[[ALLOCA:.*]] = fir.alloca !fir.array<20xi32> {bindc_name = "arr_read_write", uniq_name = "_QFcall_assumed_shape_and_size_arrayEarr_read_write"}
!HOST:%{{.*}} = arith.constant 1 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%{{.*}} = arith.constant 1 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%{{.*}} = arith.constant 10 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%[[SHAPE0:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
!HOST:%[[SLICE0:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
!HOST:%[[ARG0EMB:.*]] = fir.embox %[[ALLOCA]](%[[SHAPE0]]) [%[[SLICE0]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<10xi32>>
!HOST:%[[ARG0:.*]] = fir.convert %[[ARG0EMB]] : (!fir.box<!fir.array<10xi32>>) -> !fir.box<!fir.array<?xi32>>
!HOST:fir.call @_QMassumed_array_routinesPassumed_shape_array(%[[ARG0]]) fastmath<contract> : (!fir.box<!fir.array<?xi32>>) -> ()
!HOST:%{{.*}} = arith.constant 10 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%{{.*}} = arith.constant 1 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%{{.*}} = arith.constant 20 : i64
!HOST:%{{.*}} = fir.convert %{{.*}} : (i64) -> index
!HOST:%[[SHAPE1:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
!HOST:%[[SLICE1:.*]] = fir.slice %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.slice<1>
!HOST:%[[ARG1EMB:.*]] = fir.embox %[[ALLOCA]](%[[SHAPE1]]) [%[[SLICE1]]] : (!fir.ref<!fir.array<20xi32>>, !fir.shape<1>, !fir.slice<1>) -> !fir.box<!fir.array<11xi32>>
!HOST:%[[ADDROF:.*]] = fir.box_addr %[[ARG1EMB]] : (!fir.box<!fir.array<11xi32>>) -> !fir.ref<!fir.array<11xi32>>
!HOST:%[[ARG1:.*]] = fir.convert %[[ADDROF]] : (!fir.ref<!fir.array<11xi32>>) -> !fir.ref<!fir.array<?xi32>>
!HOST:fir.call @_QMassumed_array_routinesPassumed_size_array(%[[ARG1]]) fastmath<contract> : (!fir.ref<!fir.array<?xi32>>) -> ()
!HOST:return
!HOST:}
subroutine call_assumed_shape_and_size_array
    use assumed_array_routines
    integer :: arr_read_write(20)
    call assumed_shape_array(arr_read_write(1:10))
    call assumed_size_array(arr_read_write(10:20))
end subroutine call_assumed_shape_and_size_array
