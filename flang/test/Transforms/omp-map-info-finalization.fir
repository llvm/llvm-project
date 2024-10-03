// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

module attributes {omp.is_target_device = false} {
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
}
 
// CHECK: func.func @test_descriptor_expansion_pass(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>) {
// CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
// CHECK: %[[ALLOCA2:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
// CHECK: %[[DECLARE1:.*]]:2 = hlfir.declare %[[ARG0]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "test"} : (!fir.box<!fir.array<?xi32>>) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
// CHECK: %[[DECLARE2:.*]]:2 = hlfir.declare %[[ALLOCA2]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "test2"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
// CHECK: %[[BOUNDS:.*]] = omp.map.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK: %[[BASE_ADDR_OFF:.*]] = fir.box_offset %[[DECLARE2]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK: %[[DESC_MEMBER_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, i32) var_ptr_ptr(%[[BASE_ADDR_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP:.*]] = omp.map.info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) members(%[[DESC_MEMBER_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK: fir.store %[[DECLARE1]]#1 to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK: %[[BASE_ADDR_OFF_2:.*]] = fir.box_offset %[[ALLOCA]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK: %[[DESC_MEMBER_MAP_2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.array<?xi32>) var_ptr_ptr(%[[BASE_ADDR_OFF_2]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP_2:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(from) capture(ByRef) members(%[[DESC_MEMBER_MAP_2]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: omp.target map_entries(%[[DESC_MEMBER_MAP]] -> %[[ARG1:.*]], %[[DESC_PARENT_MAP]] -> %[[ARG2:.*]], %[[DESC_MEMBER_MAP_2]] -> %[[ARG3:.*]], %[[DESC_PARENT_MAP_2]] -> %[[ARG4:.*]] : {{.*}}) {

// -----

module attributes {omp.is_target_device = false} {
  func.func @test_derived_type_map_operand_and_block_addition(%arg0: !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) {
    %0 = hlfir.designate %arg0{"rx"}   : (!fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) -> !fir.ref<f32>
    %1 = omp.map.info var_ptr(%0 : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%rx"}
    %2 = hlfir.designate %arg0{"ry"}   : (!fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) -> !fir.ref<f32>
    %3 = omp.map.info var_ptr(%2 : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%ry"}
    %4 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>, !fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>) map_clauses(from) capture(ByRef) members(%1, %3 : [1], [4] : !fir.ref<f32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>> {name = "scalar_struct", partial_map = true}
    omp.target map_entries(%4 -> %arg1 : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) {
      omp.terminator
    }
    return
  }
}

// CHECK: func.func @test_derived_type_map_operand_and_block_addition(%{{.*}}: !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) { 
// CHECK:   %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%rx"}
// CHECK:   %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(from) capture(ByRef) -> !fir.ref<f32> {name = "scalar_struct%ry"}
// CHECK:   %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>, !fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>) map_clauses(from) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [1], [4] : !fir.ref<f32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>> {name = "scalar_struct", partial_map = true}
// CHECK:   omp.target map_entries(%[[MAP_MEMBER_1]] -> %[[ARG1:.*]], %[[MAP_MEMBER_2]] -> %[[ARG2:.*]], %[[MAP_PARENT]] -> %[[ARG3:.*]] : !fir.ref<f32>, !fir.ref<f32>, !fir.ref<!fir.type<_QFTdtype{ix:i32,rx:f32,zx:!fir.complex<4>,nested:!fir.box<!fir.heap<!fir.type<_QFTdtype>>>,ry:f32}>>) {

// -----

module attributes {omp.is_target_device = false} {
func.func @test_nested_derived_type_map_operand_and_block_addition(%arg0: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {        
    %0 = fir.declare %arg0 {uniq_name = "_QFmaptype_derived_nested_explicit_multiple_membersEsa"} : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>
    %1 = fir.field_index n, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>
    %2 = fir.coordinate_of %0, %1 : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.field) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
    %3 = fir.field_index i, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>
    %4 = fir.coordinate_of %2, %3 : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>, !fir.field) -> !fir.ref<i32>
    %5 = omp.map.info var_ptr(%4 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%n%i"}
    %6 = fir.field_index n, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>
    %7 = fir.coordinate_of %0, %6 : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.field) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>
    %8 = fir.field_index r, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>
    %9 = fir.coordinate_of %7, %8 : (!fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>>, !fir.field) -> !fir.ref<f32>
    %10 = omp.map.info var_ptr(%9 : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%n%r"}
    %11 = omp.map.info var_ptr(%0 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>, !fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>) map_clauses(tofrom) capture(ByRef) members(%5, %10 : [1,0], [1,1] : !fir.ref<i32>, !fir.ref<f32>) -> !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>> {name = "sa", partial_map = true}
    omp.target map_entries(%11 -> %arg1 : !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) {
      omp.terminator
    }
    return
  }
}

// CHECK: func.func @test_nested_derived_type_map_operand_and_block_addition(%{{.*}}: !fir.ref<!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTscalar_and_array{r:f32,n:!fir.type<_QFmaptype_derived_nested_explicit_multiple_membersTnested{i:i32,r:f32}>}>>) { 
// CHECK:   %[[MAP_MEMBER_1:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32> {name = "sa%n%i"}
// CHECK:   %[[MAP_MEMBER_2:.*]] = omp.map.info var_ptr(%{{.*}} : !fir.ref<f32>, f32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<f32> {name = "sa%n%r"}
// CHECK:   %[[MAP_PARENT:.*]] = omp.map.info var_ptr(%{{.*}} : {{.*}}, {{.*}}) map_clauses(tofrom) capture(ByRef) members(%[[MAP_MEMBER_1]], %[[MAP_MEMBER_2]] : [1,0], [1,1] : !fir.ref<i32>, !fir.ref<f32>) -> {{.*}} {name = "sa", partial_map = true}
// CHECK:   omp.target map_entries(%[[MAP_MEMBER_1]] -> %[[ARG1:.*]], %[[MAP_MEMBER_2]] -> %[[ARG2:.*]], %[[MAP_PARENT]] -> %[[ARG3:.*]] : !fir.ref<i32>, !fir.ref<f32>, {{.*}}) {
