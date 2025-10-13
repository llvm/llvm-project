// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

func.func @test_descriptor_expansion_pass(%arg0: !fir.box<!fir.array<?xi32>>) {
  %0 = fir.alloca !fir.box<!fir.heap<i32>>
  %1 = fir.zero_bits !fir.heap<i32>
  %2:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "test"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
  %3 = fir.embox %1 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %3 to %0 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %4:2 = hlfir.declare %0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "test2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
  %5 = fir.allocmem i32 {fir.must_be_heap = true}
  %6 = fir.embox %5 : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
  fir.store %6 to %4#1 : !fir.ref<!fir.box<!fir.heap<i32>>>
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 10 : index
  %dims:3 = fir.box_dims %2#1, %c1 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
  %bounds = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c2 : index) extent(%dims#1 : index) stride(%dims#2 : index) start_idx(%c0 : index) {stride_in_bytes = true}
  %7 = fir.box_addr %2#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
  %8 = omp.map.info var_ptr(%4#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>>
  %9 = omp.map.info var_ptr(%7 : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>) map_clauses(from) capture(ByRef) bounds(%bounds) -> !fir.ref<!fir.array<?xi32>>
  omp.target map_entries(%8 -> %arg1, %9 -> %arg2 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>) {
    omp.terminator
  }
  return
}

// CHECK: func.func @test_descriptor_expansion_pass(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK: %[[ALLOCA2:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
// CHECK: %[[DECLARE1:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "test"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK: %[[DECLARE2:.*]]:2 = hlfir.declare %[[ALLOCA2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "test2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK: %[[BASE_ADDR_OFF:.*]] = fir.box_offset %[[DECLARE2]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK: %[[DESC_MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(to) capture(ByRef) members(%[[DESC_MEMBER_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK: fir.store %[[DECLARE1]]#1 to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK: %[[BASE_ADDR_OFF_2:.*]] = fir.box_offset %[[ALLOCA]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK: %[[DESC_MEMBER_MAP_2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, i32) map_clauses(from) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_OFF_2]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP_2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(to) capture(ByRef) members(%[[DESC_MEMBER_MAP_2]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: omp.target map_entries(%[[DESC_PARENT_MAP]] -> %[[ARG1:.*]], %[[DESC_PARENT_MAP_2]] -> %[[ARG2:.*]], %[[DESC_MEMBER_MAP]] -> %[[ARG3:.*]], %[[DESC_MEMBER_MAP_2]] -> %[[ARG4:.*]] : {{.*}}) {

// -----

func.func @test_derived_type_map_operand_and_block_addition(%arg0: !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) {
  %0 = hlfir.designate %arg0{"rx"}   : (!fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) -> !fir.ref<f32>
  %1 = omp.map.info var_ptr(%0 : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%rx"}
  %2 = hlfir.designate %arg0{"ry"}   : (!fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) -> !fir.ref<f32>
  %3 = omp.map.info var_ptr(%2 : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%ry"}
  %4 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>, !fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>) map_clauses(from) capture(ByRef) members(%1, %3 : [1], [4] : !fir.ref<f32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>> {name = "scalar_struct", partial_map = true}
  omp.target map_entries(%4 -> %arg1 : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) {
    omp.terminator
  }
  return
}

// CHECK: func.func @test_derived_type_map_operand_and_block_addition(%{{.*}}: !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) { 
// CHECK:   %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%rx"}
// CHECK:   %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%ry"}
// CHECK:   %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>, !fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>) map_clauses(from) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [1], [4] : !fir.ref<f32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>> {name = "scalar_struct", partial_map = true}
// CHECK:   omp.target map_entries(%[[MAP_PARENT]] -> %[[ARG1:.*]], %[[MAP_MEMBER_1]] -> %[[ARG2:.*]], %[[MAP_MEMBER_2]] -> %[[ARG3:.*]] : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:complex<f32>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>, !fir.ref<f32>, !fir.ref<f32>) {

// -----

func.func @test_nested_derived_type_map_operand_and_block_addition(%arg0: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {        
  %0 = fir.declare %arg0 {uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEsa"} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>
  %2 = fir.coordinate_of %0, n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
  %4 = fir.coordinate_of %2, i : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<i32>
  %5 = omp.map.info var_ptr(%4 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%n%i"}
  %7 = fir.coordinate_of %0, n : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
  %9 = fir.coordinate_of %7, r : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>) -> !fir.ref<f32>
  %10 = omp.map.info var_ptr(%9 : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%n%r"}
  %11 = omp.map.info var_ptr(%0 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>) map_clauses(tofrom) capture(ByRef) members(%5, %10 : [1,0], [1,1] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>> {name = "sa", partial_map = true}
  omp.target map_entries(%11 -> %arg1 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {
    omp.terminator
  }
  return
}

// CHECK: func.func @test_nested_derived_type_map_operand_and_block_addition(%{{.*}}: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) { 
// CHECK:   %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%n%i"}
// CHECK:   %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%n%r"}
// CHECK:   %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [1, 0], [1, 1] : !fir.ref<i32>, !fir.ref<f32>) -> {{.*}} {name = "sa", partial_map = true}
// CHECK:   omp.target map_entries(%[[MAP_PARENT]] -> %[[ARG1:.*]], %[[MAP_MEMBER_1]] -> %[[ARG2:.*]], %[[MAP_MEMBER_2]] -> %[[ARG3:.*]] : {{.*}}, !fir.ref<i32>, !fir.ref<f32>) {

// -----

func.func @dtype_alloca_op_block_add(%arg0: !fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) {
  %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest_derived_type_allocatable_map_operand_and_block_additionEone_l"} : (!fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> (!fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>, !fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>)
  %1 = hlfir.designate %0#0{"array_j"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %2 = fir.load %1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %c0_0 = arith.constant 0 : index
  %3:3 = fir.box_dims %2, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %c1_9 = arith.constant 1 : index
  %c0_1 = arith.constant 0 : index
  %4:3 = fir.box_dims %2, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
  %c0_2 = arith.constant 0 : index
  %5 = arith.subi %4#1, %c1_9 : index
  %6 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%5 : index) extent(%4#1 : index) stride(%4#2 : index) start_idx(%3#0 : index) {stride_in_bytes = true}
  %c4 = arith.constant 4 : index
  %7 = fir.coordinate_of %0#0, array_j : (!fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
  %8 = omp.map.info var_ptr(%7 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%6) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "one_l%array_j"}
  %9 = omp.map.info var_ptr(%0#0 : !fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>, !fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>) map_clauses(tofrom) capture(ByRef) members(%8 : [4] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>> {name = "one_l", partial_map = true}
  omp.target map_entries(%9 -> %arg1 : !fir.ref<!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) {
    omp.terminator
  }
  return
}

// CHECK: func.func @dtype_alloca_op_block_add(%[[ARG0:.*]]: !fir.ref<[[REC_TY:!fir.type<_QFtest_derived_type_allocatable_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>) {
// CHECK:   %[[ALLOCA:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<[[REC_TY]]>>) -> (!fir.ref<[[REC_TY]]>>, !fir.ref<[[REC_TY]]>>)
// CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK:   %[[MEMBER_COORD:.*]] = fir.coordinate_of %[[ALLOCA]]#0, array_j : (!fir.ref<[[REC_TY]]>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:   %[[MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[MEMBER_COORD:.*]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:   %[[MAP_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
// CHECK:   %[[MAP_MEMBER_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "one_l%array_j"}
// CHECK:   %[[MAP_MEMBER_PARENT:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#0 : !fir.ref<[[REC_TY]]>>, [[REC_TY]]>) map_clauses(tofrom) capture(ByRef) members(%10, %9 : [4], [4, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<[[REC_TY]]>> {{.*}}
// CHECK:    omp.target map_entries(%[[MAP_MEMBER_PARENT]] -> %[[ARG1:.*]], %[[MAP_MEMBER_DESCRIPTOR]] -> %[[ARG2:.*]], %[[MAP_MEMBER_BASE_ADDR]] -> %[[ARG3:.*]] : !fir.ref<[[REC_TY]]>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {

// -----

func.func @alloca_dtype_map_op_block_add(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>) {
    %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_derived_type_map_operand_and_block_additionEone_l"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>)
    %1 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>
    %2 = fir.box_addr %1 : (!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>) -> !fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %3 = hlfir.designate %2{"array_j"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %4 = fir.load %3 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %c0_0 = arith.constant 0 : index
    %5:3 = fir.box_dims %4, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %6:3 = fir.box_dims %4, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c0_2 = arith.constant 0 : index
    %7 = arith.subi %6#1, %c1_0 : index
    %8 = omp.map.bounds lower_bound(%c0_2 : index) upper_bound(%7 : index) extent(%6#1 : index) stride(%6#2 : index) start_idx(%5#0 : index) {stride_in_bytes = true}
    %9 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>
    %10 = fir.coordinate_of %9, array_j : (!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %11 = omp.map.info var_ptr(%10 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%8) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "one_l%array_j"}
    %12 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>
    %13 = fir.coordinate_of %12, k : (!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>) -> !fir.ref<i32>
    %14 = omp.map.info var_ptr(%13 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "one_l%k"}
    %15 = omp.map.info var_ptr(%0#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>, !fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>) map_clauses(tofrom) capture(ByRef) members(%11, %14 : [4], [5] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>> {name = "one_l", partial_map = true}
    omp.target map_entries(%15 -> %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>>>) {
      omp.terminator
    }
    return
  }

// CHECK: func.func @alloca_dtype_map_op_block_add(%[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY:_QFtest_allocatable_derived_type_map_operand_and_block_additionTone_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>>>) {
// CHECK:     %[[ALLOCA:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_allocatable_derived_type_map_operand_and_block_additionEone_l"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>)
// CHECK:     %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index) start_idx(%{{.*}}#0 : index) {stride_in_bytes = true}
// CHECK:     %[[LOAD_ALLOCA:.*]] = fir.load %[[ALLOCA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
// CHECK:     %[[ALLOCATABLE_MEMBER_COORD:.*]] = fir.coordinate_of %[[LOAD_ALLOCA]], array_j : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:     %[[ALLOCATABLE_MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[ALLOCATABLE_MEMBER_COORD]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:     %[[MAP_ALLOCA_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[ALLOCATABLE_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[ALLOCATABLE_MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
// CHECK:     %[[MAP_ALLOCA_MEMBER_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[ALLOCATABLE_MEMBER_COORD]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
// CHECK:     %[[LOAD_ALLOCA2:.*]] = fir.load %[[ALLOCA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
// CHECK:     %[[REGULAR_MEMBER_COORD:.*]] = fir.coordinate_of %[[LOAD_ALLOCA2]], k : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) -> !fir.ref<i32>
// CHECK:     %[[MAP_REGULAR_MEMBER:.*]] = omp.map.info var_ptr(%[[REGULAR_MEMBER_COORD]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {{.*}}
// CHECK:     %[[ALLOCATABLE_PARENT_BASE_ADDR:.*]] = fir.box_offset %[[ALLOCA]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>
// CHECK:     %[[MAP_ALLOCA_PARENT_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.type<[[REC_TY]]>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[ALLOCATABLE_PARENT_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>> {{.*}}
// CHECK:     %[[MAP_PARENT_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) map_clauses(to) capture(ByRef) members(%18, %13, %12, %16 : [0], [0, 4], [0, 4, 0], [0, 5] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>> {{.*}}
// CHECK:    omp.target map_entries(%[[MAP_PARENT_DESCRIPTOR]] -> %[[ARG1:.*]], %[[MAP_ALLOCA_PARENT_BASE_ADDR]] -> %[[ARG2:.*]], %[[MAP_ALLOCA_MEMBER_DESCRIPTOR]] -> %[[ARG3:.*]], %[[MAP_ALLOCA_MEMBER_BASE_ADDR]] -> %[[ARG4:.*]], %[[MAP_REGULAR_MEMBER]] -> %[[ARG5:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) {

// -----

  func.func @alloca_nest_dtype_map_op_block_add(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>) {
    %0:2 = hlfir.declare %arg0 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_alloca_nested_derived_type_map_operand_and_block_additionEone_l"} : (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>)
    %1 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>
    %2 = fir.box_addr %1 : (!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>) -> !fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>
    %3 = hlfir.designate %2{"nest"}   : (!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) -> !fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %4 = hlfir.designate %3{"array_k"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %5 = fir.load %4 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %c0_0 = arith.constant 0 : index
    %6:3 = fir.box_dims %5, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c1_0 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %7:3 = fir.box_dims %5, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %8 = arith.subi %7#1, %c1_0 : index
    %9 = omp.map.bounds lower_bound(%c0_1 : index) upper_bound(%8 : index) extent(%7#1 : index) stride(%7#2 : index) start_idx(%6#0 : index) {stride_in_bytes = true}
    %10 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>
    %c6 = arith.constant 6 : index
    %11 = fir.coordinate_of %10, nest : (!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>) -> !fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %c2_0 = arith.constant 2 : index
    %12 = fir.coordinate_of %11, array_k : (!fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %13 = omp.map.info var_ptr(%12 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%9) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "one_l%nest%array_k"}
    %14 = fir.load %0#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>
    %15 = fir.coordinate_of %14, nest : (!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>) -> !fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %16 = fir.coordinate_of %15, k : (!fir.ref<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<i32>
    %17 = omp.map.info var_ptr(%16 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "one_l%nest%k"}
    %18 = omp.map.info var_ptr(%0#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>, !fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>) map_clauses(tofrom) capture(ByRef) members(%13, %17 : [6,2], [6,3] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>> {name = "one_l", partial_map = true}
    omp.target map_entries(%18 -> %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>>>) {
      omp.terminator
    }
    return
  }

// CHECK: func.func @alloca_nest_dtype_map_op_block_add(%[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY:_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32,nest:!fir.type<_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}>}]]>>>>) {
// CHECK:   %[[ALLOCA:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>)
// CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK:   %[[ALLOCA_LOAD:.*]] = fir.load %[[ALLOCA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
// CHECK:   %[[INTERMEDIATE_DTYPE_NESTED_MEMBER:.*]] = fir.coordinate_of %[[ALLOCA_LOAD]], nest : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) -> !fir.ref<!fir.type<[[REC_TY2:_QFtest_alloca_nested_derived_type_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>
// CHECK:   %[[NESTED_ALLOCA_MEMBER:.*]] = fir.coordinate_of %[[INTERMEDIATE_DTYPE_NESTED_MEMBER]], array_k : (!fir.ref<!fir.type<[[REC_TY2]]>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:   %[[NESTED_ALLOCA_MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[NESTED_ALLOCA_MEMBER]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:   %[[MAP_NESTED_ALLOCA_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[NESTED_ALLOCA_MEMBER]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[NESTED_ALLOCA_MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
// CHECK:   %[[MAP_NESTED_ALLOCA_MEMBER:.*]] = omp.map.info var_ptr(%[[NESTED_ALLOCA_MEMBER]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
// CHECK:   %[[ALLOCA_LOAD2:.*]] = fir.load %[[ALLOCA]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>
// CHECK:   %[[INTERMEDIATE_DTYPE_NESTED_MEMBER2:.*]] = fir.coordinate_of %[[ALLOCA_LOAD2]], nest : (!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) -> !fir.ref<!fir.type<[[REC_TY2]]>>
// CHECK:   %[[NESTED_REGULAR_MEMBER:.*]] = fir.coordinate_of %[[INTERMEDIATE_DTYPE_NESTED_MEMBER2]], k : (!fir.ref<!fir.type<[[REC_TY2]]>>) -> !fir.ref<i32>
// CHECK:   %[[MAP_NESTED_REGULAR_MEMBER:.*]] = omp.map.info var_ptr(%[[NESTED_REGULAR_MEMBER:.*]] : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {{.*}}
// CHECK:   %[[ALLOCATABLE_PARENT_BASE_ADDR:.*]] = fir.box_offset %[[ALLOCA]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>
// CHECK:   %[[MAP_ALLOCATABLE_PARENT_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.type<[[REC_TY]]>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[ALLOCATABLE_PARENT_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>> {{.*}}
// CHECK:   %[[MAP_ALLOCATABLE_PARENT_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#1 : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>) map_clauses(to) capture(ByRef) members(%21, %15, %14, %19 : [0], [0, 6, 2], [0, 6, 2, 0], [0, 6, 3] : !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) -> !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>> {{.*}}
// CHECK:    omp.target map_entries(%[[MAP_ALLOCATABLE_PARENT_DESCRIPTOR]] -> %[[ARG1:.*]], %[[MAP_ALLOCATABLE_PARENT_BASE_ADDR]] -> %[[ARG2:.*]], %[[MAP_NESTED_ALLOCA_MEMBER]] -> %[[ARG3:.*]], %[[MAP_NESTED_ALLOCA_MEMBER_BASE_ADDR]] -> %[[ARG4:.*]], %[[MAP_NESTED_REGULAR_MEMBER]] -> %[[ARG5:.*]] : !fir.ref<!fir.box<!fir.heap<!fir.type<[[REC_TY]]>>>>, !fir.llvm_ptr<!fir.ref<!fir.type<[[REC_TY]]>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, !fir.ref<i32>) {

// -----

  func.func @nest_dtype_alloca_map_op_block_add(%arg0 : !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) {
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFtest_nested_derived_type_alloca_map_operand_and_block_additionEone_l"} : (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) -> (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>, !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>)
    %1 = hlfir.designate %0#0{"nest"}   : (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) -> !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %2 = hlfir.designate %1{"array_k"}   {fortran_attrs = #fir.var_attrs<allocatable>} : (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %3 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %c0_0 = arith.constant 0 : index
    %4:3 = fir.box_dims %3, %c0_0 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c1_16 = arith.constant 1 : index
    %c0_1 = arith.constant 0 : index
    %5:3 = fir.box_dims %3, %c0_1 : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c0_18 = arith.constant 0 : index
    %6 = arith.subi %5#1, %c1_16 : index
    %7 = omp.map.bounds lower_bound(%c0_18 : index) upper_bound(%6 : index) extent(%5#1 : index) stride(%5#2 : index) start_idx(%4#0 : index) {stride_in_bytes = true}
    %c6_0 = arith.constant 6 : index
    %8 = fir.coordinate_of %0#0, nest: (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) -> !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>
    %9 = fir.coordinate_of %8, array_k : (!fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10 = omp.map.info var_ptr(%9 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%7) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "one_l%nest%array_k"}
    %11 = omp.map.info var_ptr(%0#0 : !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>, !fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>) map_clauses(tofrom) capture(ByRef) members(%10 : [6,2] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>> {name = "one_l", partial_map = true}
    omp.target map_entries(%11 -> %arg1 : !fir.ref<!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<?xi32>>>,k:i32}>}>>) {
      omp.terminator
    }
    return
  }

// CHECK:  func.func @nest_dtype_alloca_map_op_block_add(%[[ARG0:.*]]: !fir.ref<!fir.type<[[REC_TY:_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTtop_layer{i:f32,scalar:!fir.box<!fir.heap<i32>>,array_i:!fir.array<10xi32>,j:f32,array_j:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32,nest:!fir.type<_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}>}]]>>) {
// CHECK:   %[[ALLOCA:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<!fir.type<[[REC_TY]]>>) -> (!fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.type<[[REC_TY]]>>)
// CHECK:   %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK:   %[[NESTED_DTYPE_COORD:.*]] = fir.coordinate_of %[[ALLOCA]]#0, nest : (!fir.ref<!fir.type<[[REC_TY]]>>) -> !fir.ref<!fir.type<[[REC_TY2:_QFtest_nested_derived_type_alloca_map_operand_and_block_additionTmiddle_layer{i:f32,array_i:!fir.array<10xi32>,array_k:!fir.box<!fir.heap<!fir.array<\?xi32>>>,k:i32}]]>>
// CHECK:   %[[ALLOCATABLE_MEMBER:.*]] = fir.coordinate_of %[[NESTED_DTYPE_COORD]], array_k : (!fir.ref<!fir.type<[[REC_TY2]]>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:   %[[ALLOCATABLE_MEMBER_BASE_ADDR:.*]] = fir.box_offset %[[ALLOCATABLE_MEMBER]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:   %[[MAP_ALLOCATABLE_MEMBER_BASE_ADDR:.*]] = omp.map.info var_ptr(%[[ALLOCATABLE_MEMBER]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[ALLOCATABLE_MEMBER_BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
// CHECK:   %[[MAP_ALLOCATABLE_MEMBER_DESCRIPTOR:.*]] = omp.map.info var_ptr(%[[ALLOCATABLE_MEMBER]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
// CHECK:   %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%[[ALLOCA]]#0 : !fir.ref<!fir.type<[[REC_TY]]>>, !fir.type<[[REC_TY]]>) map_clauses(tofrom) capture(ByRef) members(%12, %11 : [6, 2], [6, 2, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.type<[[REC_TY]]>> {{.*}}
// CHECK:   omp.target map_entries(%[[MAP_PARENT]] -> %[[ARG1:.*]], %[[MAP_ALLOCATABLE_MEMBER_DESCRIPTOR]] -> %[[ARG2:.*]], %[[MAP_ALLOCATABLE_MEMBER_BASE_ADDR]] -> %[[ARG3:.*]] :  !fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {

// -----

  func.func @nest_dtype_mem_idx(%arg0 : !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>) {
    %0:2 = hlfir.declare %arg0 {uniq_name = "_QFmaptype_nested_derived_type_member_idxEalloca_dtype"} : (!fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>) -> (!fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>, !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>)
    %c1_15 = arith.constant 1 : index
    %1 = omp.map.bounds lower_bound(%c1_15 : index) upper_bound(%c1_15 : index) extent(%c1_15 : index) stride(%c1_15 : index) start_idx(%c1_15 : index) {stride_in_bytes = true}
    %2 = fir.coordinate_of %0#0, vertexes : (!fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>
    %3 = omp.map.bounds lower_bound(%c1_15 : index) upper_bound(%c1_15 : index) extent(%c1_15 : index) stride(%c1_15 : index) start_idx(%c1_15 : index) {stride_in_bytes = true}
    %4 = omp.map.info var_ptr(%2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>, !fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) bounds(%3) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>> {name = "alloca_dtype%vertexes(2_8)%vertexy"}
    %5 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>
    %c2_i64 = arith.constant 2 : i64
    %c1_20 = arith.constant 1 : index
    %6 = fir.convert %c2_i64 : (i64) -> index
    %7 = arith.subi %6, %c1_20 : index
    %8 = fir.coordinate_of %5, %7 : (!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>, index) -> !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
    %9 = fir.coordinate_of %8, vertexy : (!fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
    %10 = omp.map.info var_ptr(%9 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(tofrom) capture(ByRef) bounds(%1) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "alloca_dtype%vertexes(2_8)%vertexy"}
    %11 = omp.map.info var_ptr(%0#1 : !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>, !fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) members(%4, %10 : [1], [1,2] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>> {name = "alloca_dtype", partial_map = true}
    omp.target map_entries(%11 -> %arg1 : !fir.ref<!fir.type<_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>,array_i:!fir.array<10xi32>}>>) {
      omp.terminator
    }
    return
  }

// CHECK:  func.func @nest_dtype_mem_idx(%[[ARG0:.*]]: !fir.ref<!fir.type<[[REC_TY:_QFmaptype_nested_derived_type_member_idxTdtype{i:f32,vertexes:!fir.box<!fir.heap<!fir.array<\?x!fir.type<_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<\?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<\?xi32>>>}>>>>,array_i:!fir.array<10xi32>}]]>>) {
// CHECK:    %[[DECLARE:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} : (!fir.ref<!fir.type<[[REC_TY]]>>) -> (!fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.type<[[REC_TY]]>>)
// CHECK:    %[[DESC_1:.*]] = fir.coordinate_of %[[DECLARE]]#0, vertexes : (!fir.ref<!fir.type<[[REC_TY]]>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2:_QFmaptype_nested_derived_type_member_idxTvertexes{test:i32,vertexx:!fir.box<!fir.heap<!fir.array<\?xi32>>>,vertexy:!fir.box<!fir.heap<!fir.array<\?xi32>>>}]]>>>>>
// CHECK:    %[[BASE_ADDR_1:.*]] = fir.box_offset %[[DESC_1]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x!fir.type<[[REC_TY2]]>>>>
// CHECK:    %[[BASE_ADDR_MAP_1:.*]] = omp.map.info var_ptr(%[[DESC_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>, !fir.type<[[REC_TY2]]>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_1]] : !fir.llvm_ptr<!fir.ref<!fir.array<?x!fir.type<[[REC_TY2]]>>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?x!fir.type<[[REC_TY2]]>>>> {{.*}}
// CHECK:    %[[DESC_MAP_1:.*]] = omp.map.info var_ptr(%[[DESC_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>, !fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>> {{.*}}
// CHECK:    %[[DESC_LD_1:.*]] = fir.load %[[DESC_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>
// CHECK:    %[[MEMBER_ACCESS_1:.*]] = fir.coordinate_of %[[DESC_LD_1]], %{{.*}} : (!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>, index) -> !fir.ref<!fir.type<[[REC_TY2]]>>
// CHECK:    %[[DESC_2:.*]] = fir.coordinate_of %[[MEMBER_ACCESS_1]], vertexy : (!fir.ref<!fir.type<[[REC_TY2]]>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
// CHECK:    %[[BASE_ADDR_2:.*]] = fir.box_offset %[[DESC_2]] base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK:    %[[BASE_ADDR_MAP_2:.*]] = omp.map.info var_ptr(%[[DESC_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, i32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR_2]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%{{.*}}) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {{.*}}
// CHECK:    %[[DESC_MAP_2:.*]] = omp.map.info var_ptr(%[[DESC_2]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.box<!fir.heap<!fir.array<?xi32>>>) map_clauses(to) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {{.*}}
// CHECK:    %[[TOP_PARENT_MAP:.*]] = omp.map.info var_ptr(%0#1 : !fir.ref<!fir.type<[[REC_TY]]>>, !fir.type<[[REC_TY]]>) map_clauses(exit_release_or_enter_alloc) capture(ByRef) members(%6, %5, %14, %13 : [1], [1, 0], [1, 0, 2], [1, 0, 2, 0] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?x!fir.type<[[REC_TY2]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.type<[[REC_TY]]>> {{{.*}} partial_map = true}
// CHECK:    omp.target map_entries(%[[TOP_PARENT_MAP]] -> %{{.*}}, %[[DESC_MAP_1]] -> %{{.*}}, %[[BASE_ADDR_MAP_1]] -> %{{.*}}, %[[DESC_MAP_2]] -> %{{.*}}, %[[BASE_ADDR_MAP_2]] -> %{{.*}} : !fir.ref<!fir.type<[[REC_TY]]>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.type<[[REC_TY2]]>>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?x!fir.type<[[REC_TY2]]>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {

// -----

func.func @_QPreuse_alloca(%arg0: !fir.box<!fir.array<?xf64>> {fir.bindc_name = "a"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QFreuse_allocaEa"} : (!fir.box<!fir.array<?xf64>>, !fir.dscope) -> (!fir.box<!fir.array<?xf64>>, !fir.box<!fir.array<?xf64>>)
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %2:3 = fir.box_dims %1#0, %c0 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
  %c0_0 = arith.constant 0 : index
  %3 = arith.subi %2#1, %c1 : index
  %4 = omp.map.bounds lower_bound(%c0_0 : index) upper_bound(%3 : index) extent(%2#1 : index) stride(%2#2 : index) start_idx(%c1 : index) {stride_in_bytes = true}
  %5 = fir.box_addr %1#1 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
  %6 = omp.map.info var_ptr(%5 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(to) capture(ByRef) bounds(%4) -> !fir.ref<!fir.array<?xf64>> {name = "a"}
  omp.target_data map_entries(%6 : !fir.ref<!fir.array<?xf64>>) {
    %cst = arith.constant 0.000000e+00 : f64
    %c0_1 = arith.constant 0 : index
    %7 = hlfir.designate %1#0 (%c0_1)  : (!fir.box<!fir.array<?xf64>>, index) -> !fir.ref<f64>
    hlfir.assign %cst to %7 : f64, !fir.ref<f64>
    %c1_2 = arith.constant 1 : index
    %c0_3 = arith.constant 0 : index
    %8:3 = fir.box_dims %1#0, %c0_3 : (!fir.box<!fir.array<?xf64>>, index) -> (index, index, index)
    %c0_4 = arith.constant 0 : index
    %9 = arith.subi %8#1, %c1_2 : index
    %10 = omp.map.bounds lower_bound(%c0_4 : index) upper_bound(%9 : index) extent(%8#1 : index) stride(%8#2 : index) start_idx(%c1_2 : index) {stride_in_bytes = true}
    %11 = fir.box_addr %1#1 : (!fir.box<!fir.array<?xf64>>) -> !fir.ref<!fir.array<?xf64>>
    %12 = omp.map.info var_ptr(%11 : !fir.ref<!fir.array<?xf64>>, f64) map_clauses(from) capture(ByRef) bounds(%10) -> !fir.ref<!fir.array<?xf64>> {name = "a"}
    omp.target_update map_entries(%12 : !fir.ref<!fir.array<?xf64>>)
    omp.terminator
  }
  return
}

// CHECK-LABEL: @_QPreuse_alloca
// CHECK-NEXT:  %[[ALLOCA:[0-9]+]] = fir.alloca !fir.box<!fir.array<?xf64>>
// CHECK-NOT:   fir.alloca
// CHECK:       %{{[0-9]+}} = omp.map.info var_ptr(%[[ALLOCA]]
// CHECK:       %{{[0-9]+}} = omp.map.info var_ptr(%[[ALLOCA]]
// CHECK:       omp.target_data map_entries
// CHECK:         %[[BOX_OFFSET:.*]] = fir.box_offset %[[ALLOCA]]
// CHECK:         %[[LOAD_OFFSET:.*]] = fir.load %[[BOX_OFFSET]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf64>>>
// CHECK:         %{{[0-9]+}} = omp.map.info var_ptr(%[[LOAD_OFFSET]]
// CHECK:         omp.target_update map_entries
// CHECK:         omp.terminator
// CHECK:       }
// CHECK:       return


omp.private {type = firstprivate} @boxchar.privatizer : !fir.boxchar<1> copy {
  ^bb0(%arg0: !fir.boxchar<1>, %arg1: !fir.boxchar<1>):
    omp.yield(%arg0 : !fir.boxchar<1>)
  }
func.func @_QPrealtest(%arg0: !fir.boxchar<1>) {
  %0 = fir.alloca !fir.boxchar<1>
  %1 = fir.dummy_scope : !fir.dscope
  %2:2 = fir.unboxchar %arg0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %1 {uniq_name = "a0"}: (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
  fir.store %3#0 to %0 : !fir.ref<!fir.boxchar<1>>
  %4 = fir.load %0 : !fir.ref<!fir.boxchar<1>>
  %5:2 = fir.unboxchar %4 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %6:2 = fir.unboxchar %4 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  %7 = arith.subi %6#1, %c1 : index
  %8 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%7 : index) extent(%6#1 : index) stride(%c1 : index) start_idx(%c0 : index) {stride_in_bytes = true}
  %9 = omp.map.info var_ptr(%0 : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(to) capture(ByRef) bounds(%8) -> !fir.ref<!fir.boxchar<1>>
  omp.target map_entries(%9 -> %arg1 : !fir.ref<!fir.boxchar<1>>) private(@boxchar.privatizer %3#0 -> %arg2 [map_idx=0] : !fir.boxchar<1>) {
    %10:2 = fir.unboxchar %arg2 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %11:2 = hlfir.declare %10#0 typeparams %10#1 {uniq_name = "tgt_a0"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    omp.terminator
  }
  return
}


// CHECK-LABEL:   omp.private {type = firstprivate} @boxchar.privatizer : !fir.boxchar<1> copy {
// CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.boxchar<1>, %[[VAL_1:.*]]: !fir.boxchar<1>):
// CHECK:           omp.yield(%[[VAL_0]] : !fir.boxchar<1>)
// CHECK:         }

// CHECK-LABEL:   func.func @_QPrealtest(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.boxchar<1>) {
// CHECK:           %[[VAL_0:.*]] = fir.alloca !fir.boxchar<1>
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]]:2 = fir.unboxchar %[[ARG0]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]]#0 typeparams %[[VAL_2]]#1 dummy_scope %[[VAL_1]] {uniq_name = "a0"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:           fir.store %[[VAL_3]]#0 to %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
// CHECK:           %[[VAL_5:.*]]:2 = fir.unboxchar %[[VAL_4]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]]:2 = fir.unboxchar %[[VAL_4]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_8]]#1, %[[VAL_7]] : index
// CHECK:           %[[VAL_10:.*]] = omp.map.bounds lower_bound(%[[VAL_6]] : index) upper_bound(%[[VAL_9]] : index) extent(%[[VAL_8]]#1 : index) stride(%[[VAL_7]] : index) start_idx(%[[VAL_6]] : index) {stride_in_bytes = true}
// CHECK:           %[[VAL_11:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.boxchar<1>>
// CHECK:           %[[VAL_12:.*]] = fir.box_offset %[[VAL_0]] base_addr : (!fir.ref<!fir.boxchar<1>>) -> !fir.llvm_ptr<!fir.ref<!fir.char<1,?>>>
// CHECK:           %[[VAL_13:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.boxchar<1>>, !fir.char<1,?>) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%[[VAL_12]] : !fir.llvm_ptr<!fir.ref<!fir.char<1,?>>>) bounds(%[[VAL_10]]) -> !fir.ref<!fir.boxchar<1>>
// CHECK:           %[[VAL_14:.*]] = omp.map.info var_ptr(%[[VAL_0]] : !fir.ref<!fir.boxchar<1>>, !fir.boxchar<1>) map_clauses(to) capture(ByRef) members(%[[VAL_13]] : [0] : !fir.ref<!fir.boxchar<1>>) -> !fir.ref<!fir.boxchar<1>>
// CHECK:           omp.target map_entries(%[[VAL_14]] -> %[[VAL_15:.*]], %[[VAL_13]] -> %[[VAL_16:.*]] : !fir.ref<!fir.boxchar<1>>, !fir.ref<!fir.boxchar<1>>) private(@boxchar.privatizer %[[VAL_3]]#0 -> %[[VAL_17:.*]] [map_idx=0] : !fir.boxchar<1>) {
// CHECK:             %[[VAL_18:.*]]:2 = fir.unboxchar %[[VAL_17]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
// CHECK:             %[[VAL_19:.*]]:2 = hlfir.declare %[[VAL_18]]#0 typeparams %[[VAL_18]]#1 {uniq_name = "tgt_a0"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
