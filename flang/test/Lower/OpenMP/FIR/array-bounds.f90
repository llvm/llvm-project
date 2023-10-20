!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes HOST
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-target-device %s -o - | FileCheck %s --check-prefixes DEVICE

!DEVICE-LABEL: func.func @_QPread_write_section_omp_outline_0(
!DEVICE-SAME: %[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPread_write_section"} {
!DEVICE:  %[[C1:.*]] = arith.constant 1 : index
!DEVICE:  %[[C2:.*]] = arith.constant 4 : index
!DEVICE:  %[[C3:.*]] = arith.constant 1 : index
!DEVICE:  %[[C4:.*]] = arith.constant 1 : index
!DEVICE:  %[[BOUNDS0:.*]] = omp.bounds   lower_bound(%[[C1]] : index) upper_bound(%[[C2]] : index) stride(%[[C4]] : index) start_idx(%[[C4]] : index)
!DEVICE:  %[[MAP0:.*]] = omp.map_info var_ptr(%[[ARG1]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!DEVICE:  %[[C5:.*]] = arith.constant 1 : index
!DEVICE:  %[[C6:.*]] = arith.constant 4 : index
!DEVICE:  %[[C7:.*]] = arith.constant 1 : index
!DEVICE:  %[[C8:.*]] = arith.constant 1 : index
!DEVICE:  %[[BOUNDS1:.*]] = omp.bounds   lower_bound(%[[C5]] : index) upper_bound(%[[C6]] : index) stride(%[[C8]] : index) start_idx(%[[C8]] : index)
!DEVICE:  %[[MAP1:.*]] = omp.map_info var_ptr(%[[ARG2]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!DEVICE:  omp.target   map_entries(%[[MAP0]], %[[MAP1]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>) {

!HOST-LABEL:  func.func @_QPread_write_section() {
!HOST:  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFread_write_sectionEi"}
!HOST:  %[[READ:.*]] = fir.address_of(@_QFread_write_sectionEsp_read) : !fir.ref<!fir.array<10xi32>>
!HOST:  %[[WRITE:.*]] = fir.address_of(@_QFread_write_sectionEsp_write) : !fir.ref<!fir.array<10xi32>>
!HOST:  %[[C1:.*]] = arith.constant 1 : index
!HOST:  %[[C2:.*]] = arith.constant 1 : index
!HOST:  %[[C3:.*]] = arith.constant 4 : index
!HOST:  %[[BOUNDS0:.*]] = omp.bounds   lower_bound(%[[C2]] : index) upper_bound(%[[C3]] : index) stride(%[[C1]] : index) start_idx(%[[C1]] : index)
!HOST:  %[[MAP0:.*]] = omp.map_info var_ptr(%[[READ]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!HOST:  %[[C4:.*]] = arith.constant 1 : index
!HOST:  %[[C5:.*]] = arith.constant 1 : index
!HOST:  %[[C6:.*]] = arith.constant 4 : index
!HOST:  %[[BOUNDS1:.*]] = omp.bounds   lower_bound(%[[C5]] : index) upper_bound(%[[C6]] : index) stride(%[[C4]] : index) start_idx(%[[C4]] : index)
!HOST:  %[[MAP1:.*]] = omp.map_info var_ptr(%[[WRITE]] : !fir.ref<!fir.array<10xi32>>, !fir.array<10xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!HOST:  omp.target   map_entries(%[[MAP0]], %[[MAP1]] : !fir.ref<!fir.array<10xi32>>, !fir.ref<!fir.array<10xi32>>) {

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
!DEVICE-LABEL: func.func @_QMassumed_array_routinesPassumed_shape_array_omp_outline_0(
!DEVICE-SAME: %[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QMassumed_array_routinesPassumed_shape_array"} {
!DEVICE: %[[C0:.*]] = arith.constant 1 : index
!DEVICE: %[[C1:.*]] = arith.constant 4 : index
!DEVICE: %[[C2:.*]] = arith.constant 0 : index
!DEVICE: %[[C3:.*]]:3 = fir.box_dims %[[ARG1]], %[[C2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!DEVICE: %[[C4:.*]] = arith.constant 1 : index
!DEVICE: %[[BOUNDS:.*]] = omp.bounds   lower_bound(%[[C0]] : index) upper_bound(%[[C1]] : index) stride(%[[C3]]#2 : index) start_idx(%[[C4]] : index) {stride_in_bytes = true}
!DEVICE: %[[ARGADDR:.*]] = fir.box_addr %[[ARG1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
!DEVICE: %[[MAP:.*]] = omp.map_info var_ptr(%[[ARGADDR]] : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!DEVICE: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<?xi32>>) {

!HOST-LABEL: func.func @_QMassumed_array_routinesPassumed_shape_array(
!HOST-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
!HOST: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_shape_arrayEi"}
!HOST: %[[C0:.*]] = arith.constant 1 : index
!HOST: %[[C1:.*]] = arith.constant 0 : index
!HOST: %[[C2:.*]]:3 = fir.box_dims %arg0, %[[C1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
!HOST: %[[C3:.*]] = arith.constant 1 : index
!HOST: %[[C4:.*]] = arith.constant 4 : index
!HOST: %[[BOUNDS:.*]] = omp.bounds   lower_bound(%[[C3]] : index) upper_bound(%[[C4]] : index) stride(%[[C2]]#2 : index) start_idx(%[[C0]] : index) {stride_in_bytes = true}
!HOST: %[[ADDROF:.*]] = fir.box_addr %arg0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
!HOST: %[[MAP:.*]] = omp.map_info var_ptr(%[[ADDROF]] : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!HOST: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<?xi32>>) {
    subroutine assumed_shape_array(arr_read_write)
            integer, intent(inout) :: arr_read_write(:)

        !$omp target map(tofrom:arr_read_write(2:5))
            do i = 2, 5
                arr_read_write(i) = i
            end do
        !$omp end target
        end subroutine assumed_shape_array

!DEVICE-LABEL:   func.func @_QMassumed_array_routinesPassumed_size_array_omp_outline_0(
!DEVICE-SAME:    %[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<!fir.array<?xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QMassumed_array_routinesPassumed_size_array"} {
!DEVICE: %[[C0:.*]] = arith.constant 1 : index
!DEVICE: %[[C1:.*]] = arith.constant 4 : index
!DEVICE: %[[C2:.*]] = arith.constant 1 : index
!DEVICE: %[[C3:.*]] = arith.constant 1 : index
!DEVICE: %[[BOUNDS:.*]] = omp.bounds   lower_bound(%[[C0]] : index) upper_bound(%[[C1]] : index) stride(%[[C3]] : index) start_idx(%[[C3]] : index)
!DEVICE: %[[MAP:.*]] = omp.map_info var_ptr(%[[ARG1]] : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!DEVICE: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<?xi32>>) {

!HOST-LABEL: func.func @_QMassumed_array_routinesPassumed_size_array(
!HOST-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr_read_write"}) {
!HOST: %[[ALLOCA:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QMassumed_array_routinesFassumed_size_arrayEi"}
!HOST: %[[C0:.*]] = arith.constant 1 : index
!HOST: %[[C1:.*]] = arith.constant 1 : index
!HOST: %[[C2:.*]] = arith.constant 4 : index
!HOST: %[[BOUNDS:.*]]  = omp.bounds   lower_bound(%[[C1]] : index) upper_bound(%[[C2]] : index) stride(%[[C0]] : index) start_idx(%[[C0]] : index)
!HOST: %[[MAP:.*]] = omp.map_info var_ptr(%[[ARG0]] : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>)   map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.ref<!fir.array<?xi32>> {name = "arr_read_write(2:5)"}
!HOST: omp.target   map_entries(%[[MAP]] : !fir.ref<!fir.array<?xi32>>) {
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
