// RUN: fir-opt --split-input-file --omp-map-info-finalization %s | FileCheck %s

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  func.func @test_polymorphic_base_addr_runtime_size(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_polymorphic_base_addr_runtime_size
// CHECK-SAME:  (%[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
// CHECK:         %[[BASE_ADDR:.*]] = fir.box_offset %[[ARG0]] base_addr : (!fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
// CHECK:         %[[BOX:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>
// CHECK:         %[[ELEM_SIZE:.*]] = fir.box_elesize %[[BOX]] : (!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) -> index
// CHECK:         %[[BYTE_OFFSET:.*]] = arith.muli %{{.*}}, %[[ELEM_SIZE]] : index
// CHECK:         %[[BYTE_EXTENT:.*]] = arith.muli %{{.*}}, %[[ELEM_SIZE]] : index
// CHECK:         %[[BYTE_END:.*]] = arith.addi %[[BYTE_OFFSET]], %[[BYTE_EXTENT]] : index
// CHECK:         %[[UPPER:.*]] = arith.subi %[[BYTE_END]], %{{.*}} : index
// CHECK:         %[[BOUND:.*]] = omp.map.bounds lower_bound(%[[BYTE_OFFSET]] : index) upper_bound(%[[UPPER]] : index) extent(%[[BYTE_EXTENT]] : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) stride_in_bytes(true)
// CHECK:         %[[BASE_MAP:.*]] = omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>, i8) bounds(%[[BOUND]]) name("") -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
// CHECK:         omp.map.info {{.*}} members(%[[BASE_MAP]] : [0] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>) name("obj")

// -----

builtin.module attributes {omp.version = #omp.version<version = 60>} {
  func.func @test_polymorphic_base_addr_static_size_pre_61(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) name("obj") -> !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_polymorphic_base_addr_static_size_pre_61
// CHECK-SAME:  (%[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
// CHECK:         %[[BASE_ADDR:.*]] = fir.box_offset %[[ARG0]] base_addr : (!fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
// CHECK-NOT:     fir.box_elesize
// CHECK-NOT:     omp.map.bounds
// CHECK:         omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>, !fir.type<base_t{base:i32}>) name("")
