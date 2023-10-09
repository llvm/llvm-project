
!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine map_pointer()
    integer,  pointer :: map_ptr(:)     
    allocate(map_ptr(10))
    !CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "map_ptr", uniq_name = "_QFmap_pointerEmap_ptr"}
    !CHECK: %[[DESC:.*]]:2 = hlfir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFmap_pointerEmap_ptr"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
    !CHECK: %[[LOAD_FROM_DESC:.*]] = fir.load %[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    !CHECK: %[[MAP_BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
    !CHECK: %[[MAP_DESC:.*]] = omp.map_info var_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)   map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "map_ptr"}
    !CHECK: %[[PTR_ADDR:.*]] = fir.box_addr %[[LOAD_FROM_DESC]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
    !CHECK: %[[MAP_PTR:.*]] = omp.map_info var_ptr(%[[PTR_ADDR]] : !fir.ptr<!fir.array<?xi32>>)   var_ptr_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[MAP_BOUNDS]]) -> !fir.ptr<!fir.array<?xi32>> {name = "map_ptr"}
    !CHECK: omp.target   map_entries(%[[MAP_DESC]], %[[MAP_PTR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ptr<!fir.array<?xi32>>) {
    !$omp target map(tofrom: map_ptr) 
    !$omp end target
end subroutine map_pointer

subroutine map_alloca()
    integer,  allocatable :: map_al(:) 
    allocate(map_al(10)) 
    !CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>> {bindc_name = "map_al", uniq_name = "_QFmap_allocaEmap_al"}
    !CHECK: %[[DESC:.*]]:2 = hlfir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFmap_allocaEmap_al"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
    !CHECK: %[[LOAD_FROM_DESC:.*]] = fir.load %[[DESC]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    !CHECK: %[[MAP_BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
    !CHECK: %[[MAP_DESC:.*]] = omp.map_info var_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)   map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "map_al"}
    !CHECK: %[[PTR_ADDR:.*]] = fir.box_addr %[[LOAD_FROM_DESC]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
    !CHECK: %[[MAP_PTR:.*]] = omp.map_info var_ptr(%[[PTR_ADDR]] : !fir.heap<!fir.array<?xi32>>)   var_ptr_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[MAP_BOUNDS]]) -> !fir.heap<!fir.array<?xi32>> {name = "map_al"}
    !CHECK: omp.target   map_entries(%[[MAP_DESC]], %[[MAP_PTR]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.heap<!fir.array<?xi32>>) {
    !$omp target map(tofrom: map_al) 
    !$omp end target
end subroutine map_alloca

subroutine map_pointer_target()
    integer,  pointer :: a(:)
    integer, target :: b(10)
    a => b
    !CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "a", uniq_name = "_QFmap_pointer_targetEa"}
    !CHECK: %[[DESC:.*]]:2 = hlfir.declare %[[ALLOCA]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFmap_pointer_targetEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
    !CHECK: %[[LOAD_FROM_DESC:.*]] = fir.load %[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    !CHECK: %[[MAP_BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
    !CHECK: %[[MAP_DESC:.*]] = omp.map_info var_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)   map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "a"}
    !CHECK: %[[PTR_ADDR:.*]] = fir.box_addr %[[LOAD_FROM_DESC]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
    !CHECK: %[[MAP_PTR:.*]] = omp.map_info var_ptr(%[[PTR_ADDR]] : !fir.ptr<!fir.array<?xi32>>)   var_ptr_ptr(%[[DESC]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[MAP_BOUNDS]]) -> !fir.ptr<!fir.array<?xi32>> {name = "a"}
    !CHECK: omp.target   map_entries(%[[MAP_DESC]], %[[MAP_PTR]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ptr<!fir.array<?xi32>>) {
    !$omp target map(tofrom: a) 
    !$omp end target
end subroutine map_pointer_target

subroutine map_pointer_target_section()
    integer,target  :: A(30)
    integer,pointer :: p(:)
    !CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.array<30xi32> {bindc_name = "a", fir.target, uniq_name = "_QFmap_pointer_target_sectionEa"}
    !CHECK: %[[SHAPE:.*]] = fir.shape %c30 : (index) -> !fir.shape<1>
    !CHECK: %[[DESC_1:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFmap_pointer_target_sectionEa"} : (!fir.ref<!fir.array<30xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<30xi32>>, !fir.ref<!fir.array<30xi32>>)
    !CHECK: %[[ALLOCA_2:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>> {bindc_name = "p", uniq_name = "_QFmap_pointer_target_sectionEp"}
    !CHECK: %[[DESC_2:.*]]:2 = hlfir.declare %[[ALLOCA_2]] {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFmap_pointer_target_sectionEp"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
    !CHECK: %[[MAP_1_BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) stride({{.*}}) start_idx({{.*}})
    !CHECK: %[[MAP_1:.*]] = omp.map_info var_ptr(%[[DESC_1]]#1 : !fir.ref<!fir.array<30xi32>>)   map_clauses(tofrom) capture(ByRef) bounds(%[[MAP_1_BOUNDS]]) -> !fir.ref<!fir.array<30xi32>> {name = "a(1:4)"}
    !CHECK: omp.target_data   map_entries(%[[MAP_1]] : !fir.ref<!fir.array<30xi32>>) {
    !$omp target data map( A(1:4) )
        p=>A
        !CHECK: %[[LOAD:.*]] = fir.load %[[DESC_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
        !CHECK: %[[MAP_3_BOUNDS:.*]] = omp.bounds   lower_bound({{.*}}) upper_bound({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
        !CHECK: %[[MAP_2:.*]] = omp.map_info var_ptr(%[[DESC_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)   map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "p(8:27)"}
        !CHECK: %[[MAP_ADDR_OF:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
        !CHECK: %[[MAP_3:.*]] = omp.map_info var_ptr(%[[MAP_ADDR_OF]] : !fir.ptr<!fir.array<?xi32>>)   var_ptr_ptr(%[[DESC_2]]#1 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[MAP_3_BOUNDS]]) -> !fir.ptr<!fir.array<?xi32>> {name = "p(8:27)"}
        !CHECK: omp.target   map_entries(%[[MAP_2]], %[[MAP_3]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ptr<!fir.array<?xi32>>) {
        !$omp target map( p(8:27) )
        A(3) = 0
        p(9) = 0
        !$omp end target
    !$omp end target data
end subroutine map_pointer_target_section
