// RUN: fir-opt --omp-descriptor-map-info-gen %s | FileCheck %s

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
    %bounds = omp.bounds lower_bound(%c1 : index) upper_bound(%c2 : index) extent(%dims#1 : index) stride(%dims#2 : index) start_idx(%c0 : index) {stride_in_bytes = true}
    %7 = fir.box_addr %2#1 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
    %8 = omp.map_info var_ptr(%4#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.box<!fir.heap<i32>>>
    %9 = omp.map_info var_ptr(%7 : !fir.ref<!fir.array<?xi32>>, !fir.array<?xi32>) map_clauses(from) capture(ByRef) bounds(%bounds) -> !fir.ref<!fir.array<?xi32>>
    omp.target map_entries(%8 -> %arg1, %9 -> %arg2 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.array<?xi32>>) {
    ^bb0(%arg1: !fir.ref<!fir.box<!fir.heap<i32>>>, %arg2: !fir.ref<!fir.array<?xi32>>):
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
// CHECK: %[[BOUNDS:.*]] = omp.bounds lower_bound(%{{.*}} : index) upper_bound(%{{.*}} : index) extent(%{{.*}} : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) {stride_in_bytes = true}
// CHECK: %[[BASE_ADDR_OFF:.*]] = fir.box_offset %[[DECLARE2]]#1 base_addr : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.llvm_ptr<!fir.ref<i32>>
// CHECK: %[[DESC_MEMBER_MAP:.*]] = omp.map_info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, i32) var_ptr_ptr(%[[BASE_ADDR_OFF]] : !fir.llvm_ptr<!fir.ref<i32>>) map_clauses(tofrom) capture(ByRef) -> !fir.llvm_ptr<!fir.ref<i32>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP:.*]] = omp.map_info var_ptr(%[[DECLARE2]]#1 : !fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.heap<i32>>) map_clauses(tofrom) capture(ByRef) members(%[[DESC_MEMBER_MAP]] : !fir.llvm_ptr<!fir.ref<i32>>) -> !fir.ref<!fir.box<!fir.heap<i32>>>
// CHECK: fir.store %[[DECLARE1]]#1 to %[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
// CHECK: %[[BASE_ADDR_OFF_2:.*]] = fir.box_offset %[[ALLOCA]] base_addr : (!fir.ref<!fir.box<!fir.array<?xi32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
// CHECK: %[[DESC_MEMBER_MAP_2:.*]] = omp.map_info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.array<?xi32>) var_ptr_ptr(%[[BASE_ADDR_OFF_2]] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) map_clauses(from) capture(ByRef) bounds(%[[BOUNDS]]) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>> {name = ""}
// CHECK: %[[DESC_PARENT_MAP_2:.*]] = omp.map_info var_ptr(%[[ALLOCA]] : !fir.ref<!fir.box<!fir.array<?xi32>>>, !fir.box<!fir.array<?xi32>>) map_clauses(from) capture(ByRef) members(%15 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.array<?xi32>>
// CHECK: omp.target map_entries(%[[DESC_MEMBER_MAP]] -> %[[ARG1:.*]], %[[DESC_PARENT_MAP]] -> %[[ARG2:.*]], %[[DESC_MEMBER_MAP_2]] -> %[[ARG3:.*]], %[[DESC_PARENT_MAP_2]] -> %[[ARG4:.*]] : {{.*}}) {
// CHECK: ^bb0(%[[ARG1]]: !fir.llvm_ptr<!fir.ref<i32>>, %[[ARG2]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[ARG3]]: !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>, %[[ARG4]]: !fir.ref<!fir.array<?xi32>>):
