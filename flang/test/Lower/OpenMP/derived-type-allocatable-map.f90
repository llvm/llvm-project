!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.type<[[ONE_LAYER_TY:_QFdtype_alloca_map_op_blockTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]> {{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {{{.*}}} : (!fir.ref<!fir.type<[[ONE_LAYER_TY]]>>) -> (!fir.ref<!fir.type<[[ONE_LAYER_TY]]>>, !fir.ref<!fir.type<[[ONE_LAYER_TY]]>>)
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
!CHECK: %[[MEMBER_INDEX:.*]] = arith.constant 4 : index
!CHECK: %[[MEMBER_COORD:.*]] = fir.coordinate_of %[[DECLARE]]#0, %[[MEMBER_INDEX]] : (!fir.ref<!fir.type<[[ONE_LAYER_TY]]>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[MEMBER_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK: %[[MAP_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) var_ptr_ptr(%[[MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
!CHECK: %[[MAP_MEMBER_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
!CHECK: %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<[[ONE_LAYER_TY]]>>, !fir.type<[[ONE_LAYER_TY]]>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_DESCRIPTOR]], %[[MAP_MEMBER_BASE_ADDR]] : [4], [4, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.type<[[ONE_LAYER_TY]]>> {{{.*}} partial_map = true}
!CHECK:   omp.target map_entries(%[[MAP_PARENT]] -> %[[ARG0:.*]], %[[MAP_MEMBER_DESCRIPTOR]] -> %[[ARG1:.*]], %[[MAP_MEMBER_BASE_ADDR]] -> %[[ARG2:.*]] : !fir.ref<!fir.type<[[ONE_LAYER_TY]]>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:        %{{.*}}:2 = hlfir.declare %[[ARG0]] {{{.*}}} : (!fir.ref<!fir.type<[[ONE_LAYER_TY]]>>) -> (!fir.ref<!fir.type<[[ONE_LAYER_TY]]>>, !fir.ref<!fir.type<[[ONE_LAYER_TY]]>>)
subroutine dtype_alloca_map_op_block()
    type :: one_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    end type one_layer

    type(one_layer) :: one_l

    allocate(one_l%array_j(10))

    !$omp target map(tofrom: one_l%array_j)
        one_l%array_j(1) = 10
    !$omp end target
end subroutine

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<[[REC_TY:_QFalloca_dtype_op_block_addTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>> {{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {{{.*}}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>)
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
!CHECK: %[[LOAD_DTYPE:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
!CHECK: %[[MEMBER_INDEX:.*]] = arith.constant 4 : index
!CHECK: %[[MEMBER_COORD:.*]] = fir.coordinate_of %[[LOAD_DTYPE]], %[[MEMBER_INDEX]] : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[MEMBER_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK: %[[MAP_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) var_ptr_ptr(%[[MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
!CHECK: %[[MAP_MEMBER_DESC:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
!CHECK: %[[LOAD_DTYPE:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
!CHECK: %[[MEMBER_COORD:.*]] = arith.constant 5 : index
!CHECK: %[[REGULAR_MEMBER:.*]] = fir.coordinate_of %[[LOAD_DTYPE]], %[[MEMBER_COORD]] : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>, index) -> !fir.ref<i32>
!CHECK: %[[MAP_REGULAR_MEMBER:.*]] = omp.map.info var_ptr(%[[REGULAR_MEMBER]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {{.*}}
!CHECK: %[[DTYPE_BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>
!CHECK: %[[MAP_DTYPE_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.type<[[REC_TY]]>) var_ptr_ptr(%[[DTYPE_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>> {{.*}}
!CHECK: %[[MAP_DTYPE_DESC:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) map_clauses(to) capture(ByRef) members(%[[MAP_DTYPE_BASE_ADDR]], %[[MAP_MEMBER_DESC]], %[[MAP_MEMBER_BASE_ADDR]], %[[MAP_REGULAR_MEMBER]] : [0], [0, 4], [0, 4, 0], [0, 5] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>> {{.*}}
!CHECK: omp.target map_entries(%[[MAP_DTYPE_DESC]] -> %[[ARG0:.*]], %[[MAP_DTYPE_BASE_ADDR]] -> %[[ARG1:.*]], %[[MAP_MEMBER_DESC]] -> %[[ARG2:.*]], %[[MAP_MEMBER_BASE_ADDR]] -> %[[ARG3:.*]], %[[MAP_REGULAR_MEMBER]] -> %[[ARG4:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) {
!CHECK:  %{{.*}}:2 = hlfir.declare %[[ARG0]] {{{.*}}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>)
subroutine alloca_dtype_op_block_add()
    type :: one_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    end type one_layer

    type(one_layer), allocatable :: one_l

    allocate(one_l)
    allocate(one_l%array_j(10))

    !$omp target map(tofrom: one_l%array_j, one_l%k)
        one_l%array_j(1) = 10
        one_l%k = 20
    !$omp end target
end subroutine

!CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<!fir.type<[[REC_TY:_QFalloca_nest_dype_map_op_block_addTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32,nest:!fir.type<_QFalloca_nest_dype_map_op_block_addTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>}>>> {{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>)
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
!CHECK: %[[LOAD:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>
!CHECK: %[[NESTED_DTYPE_INDEX:.*]] = arith.constant 6 : index
!CHECK: %[[NESTED_DTYPE_COORD:.*]] = fir.coordinate_of %[[LOAD]], %[[NESTED_DTYPE_INDEX]] : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>, index) -> !fir.ref<!fir.type<[[REC_TY2:_QFalloca_nest_dype_map_op_block_addTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>
!CHECK: %[[NESTED_MEMBER_INDEX:.*]] = arith.constant 2 : index
!CHECK: %[[NESTED_MEMBER_COORD:.*]] = fir.coordinate_of %[[NESTED_DTYPE_COORD]], %[[NESTED_MEMBER_INDEX]] : (!fir.ref<!fir.type<[[REC_TY2]]>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[NESTED_MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[NESTED_MEMBER_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK: %[[MAP_NESTED_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[NESTED_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) var_ptr_ptr(%[[NESTED_MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
!CHECK: %[[MAP_NESTED_MEMBER_COORD:.*]] = omp.map.info var_ptr(%[[NESTED_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
!CHECK: %[[LOAD:.*]] = fir.load %[[DECLARE]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>
!CHECK: %[[NESTED_DTYPE_INDEX:.*]] = arith.constant 6 : index
!CHECK: %[[NESTED_DTYPE_COORD:.*]] = fir.coordinate_of %[[LOAD]], %[[NESTED_DTYPE_INDEX]] : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>, index) -> !fir.ref<!fir.type<[[REC_TY2]]>>
!CHECK: %[[NESTED_MEMBER_INDEX:.*]] = arith.constant 3 : index
!CHECK: %[[REGULAR_NESTED_MEMBER_COORD:.*]] = fir.coordinate_of %[[NESTED_DTYPE_COORD]], %[[NESTED_MEMBER_INDEX]] : (!fir.ref<!fir.type<[[REC_TY2]]>>, index) -> !fir.ref<i32>
!CHECK: %[[MAP_REGULAR_NESTED_MEMBER:.*]] = omp.map.info var_ptr(%[[REGULAR_NESTED_MEMBER_COORD]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {{.*}}
!CHECK: %[[DTYPE_BASE_ADDR:.*]] = fir.box_offset %[[DECLARE]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>}>>>
!CHECK: %[[MAP_DTYPE_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>, !fir.type<[[REC_TY]]>}>) var_ptr_ptr(%[[DTYPE_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>}>>>) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>}>>> {{.*}}
!CHECK: %[[MAP_DTYPE:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>, !fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>) map_clauses(to) capture(ByRef) members(%[[MAP_DTYPE_BASE_ADDR]], %[[MAP_NESTED_MEMBER_COORD]], %[[MAP_NESTED_MEMBER_BASE_ADDR]], %[[MAP_REGULAR_NESTED_MEMBER]] : [0], [0, 6, 2], [0, 6, 2, 0], [0, 6, 3] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>}>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>> {{.*}}
!CHECK: omp.target map_entries(%[[MAP_DTYPE]] -> %[[ARG0:.*]], %[[MAP_DTYPE_BASE_ADDR]] -> %[[ARG1:.*]], %[[MAP_NESTED_MEMBER_COORD]] -> %[[ARG2:.*]], %[[MAP_NESTED_MEMBER_BASE_ADDR]] -> %[[ARG3:.*]], %[[MAP_REGULAR_NESTED_MEMBER]] -> %[[ARG4:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>, !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>}>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) {
!CHECK:  %{{.*}}:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>}>>>>)
subroutine alloca_nest_dype_map_op_block_add()
    type :: middle_layer
    real(4) :: i
    integer(4) :: array_i(10)
    integer, allocatable :: array_k(:)
    integer(4) :: k
    end type middle_layer

    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(middle_layer) :: nest
    end type top_layer

    type(top_layer), allocatable :: one_l

    allocate(one_l)
    allocate(one_l%nest%array_k(10))

    !$omp target map(tofrom: one_l%nest%array_k, one_l%nest%k)
        one_l%nest%array_k(1) = 10
        one_l%nest%k = 20
    !$omp end target
end subroutine

!CHECK: %[[ALLOCA]] = fir.alloca !fir.type<[[REC_TY:_QFnest_dtype_alloca_map_op_block_addTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32,nest:!fir.type<_QFnest_dtype_alloca_map_op_block_addTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}>}]]> {{.*}}
!CHECK: %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA:.*]] {{.*}} : (!fir.ref<!fir.type<[[REC_TY]]>>) -> (!fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.type<[[REC_TY]]>>)
!CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound({{.*}}) upper_bound({{.*}}) extent({{.*}}) stride({{.*}}) start_idx({{.*}}) {stride_in_bytes = true}
!CHECK: %[[NESTED_DTYPE_INDEX:.*]] = arith.constant 6 : index
!CHECK: %[[NESTED_DTYPE_COORD:.*]] = fir.coordinate_of %[[DECLARE]]#0, %[[NESTED_DTYPE_INDEX]] : (!fir.ref<!fir.type<[[REC_TY]]>>, index) -> !fir.ref<!fir.type<[[REC_TY2:_QFnest_dtype_alloca_map_op_block_addTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>
!CHECK: %[[NESTED_MEMBER_INDEX:.*]] = arith.constant 2 : index
!CHECK: %[[NESTED_MEMBER_COORD:.*]] = fir.coordinate_of %[[NESTED_DTYPE_COORD]], %[[NESTED_MEMBER_INDEX]] : (!fir.ref<!fir.type<[[REC_TY2]]>>, index) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
!CHECK: %[[NESTED_MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[NESTED_MEMBER_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
!CHECK: %[[MAP_NESTED_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[NESTED_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) var_ptr_ptr(%[[NESTED_MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
!CHECK: %[[MAP_NESTED_MEMBER_DESC:.*]] = omp.map.info var_ptr(%[[NESTED_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
!CHECK: %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%[[DECLARE]]#1 : !fir.ref<!fir.type<[[REC_TY]]>>, !fir.type<[[REC_TY]]>) map_clauses(tofrom) capture(ByRef) members(%[[MAP_NESTED_MEMBER_DESC]], %[[MAP_NESTED_MEMBER_BASE_ADDR]] : [6, 2], [6, 2, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.type<[[REC_TY]]>> {{.*}}
!CHECK: omp.target map_entries(%[[MAP_PARENT]] -> %[[ARG0:.*]], %[[MAP_NESTED_MEMBER_DESC]] -> %[[ARG1:.*]], %[[MAP_NESTED_MEMBER_BASE_ADDR]] -> %[[ARG2:.*]] : !fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
!CHECK:  %{{.*}}:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<!fir.type<[[REC_TY]]>>) -> (!fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.type<[[REC_TY]]>>)
subroutine nest_dtype_alloca_map_op_block_add()
    type :: middle_layer
    real(4) :: i
    integer(4) :: array_i(10)
    integer, allocatable :: array_k(:)
    integer(4) :: k
    end type middle_layer

    type :: top_layer
    real(4) :: i
    integer, allocatable :: scalar
    integer(4) :: array_i(10)
    real(4) :: j
    integer, allocatable :: array_j(:)
    integer(4) :: k
    type(middle_layer) :: nest
    end type top_layer

    type(top_layer) :: one_l

    allocate(one_l%nest%array_k(10))

    !$omp target map(tofrom: one_l%nest%array_k)
        one_l%nest%array_k(1) = 25
    !$omp end target
end subroutine
