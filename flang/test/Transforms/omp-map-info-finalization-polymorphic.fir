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
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         %[[BOX:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>
// CHECK:         %[[ELEM_SIZE:.*]] = fir.box_elesize %[[BOX]] : (!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) -> index
// CHECK:         %[[BYTE_EXTENT:.*]] = arith.muli %[[ONE]], %[[ELEM_SIZE]] : index
// CHECK:         %[[BYTE_OFFSET:.*]] = arith.addi %[[ZERO]], %[[BYTE_EXTENT]] : index
// CHECK:         %[[UPPER:.*]] = arith.subi %[[BYTE_OFFSET]], %[[ONE]] : index
// CHECK:         %[[BOUND:.*]] = omp.map.bounds lower_bound(%[[ZERO]] : index) upper_bound(%[[UPPER]] : index) extent(%[[BYTE_EXTENT]] : index) stride(%[[ONE]] : index) start_idx(%[[ZERO]] : index) stride_in_bytes(true)
// CHECK:         %[[BASE_MAP:.*]] = omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>, i8) bounds(%[[BOUND]]) name("obj") -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
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
// CHECK:         omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>, !fir.type<base_t{base:i32}>) name("obj")

// -----

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  omp.declare_mapper @base_t_mapper : !fir.type<base_t{base:i32}> {
  ^bb0(%arg0: !fir.ref<!fir.type<base_t{base:i32}>>):
    omp.declare_mapper.info
  }

  func.func @test_polymorphic_base_addr_with_mapper_declared_size(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) mapper(@base_t_mapper) name("obj") -> !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_polymorphic_base_addr_with_mapper_declared_size
// CHECK-SAME:  (%[[ARG0:.*]]: !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) {
// CHECK:         %[[BASE_ADDR:.*]] = fir.box_offset %[[ARG0]] base_addr : (!fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
// CHECK-NOT:     fir.box_elesize
// CHECK-NOT:     omp.map.bounds
// CHECK:         omp.map.info var_ptr(%[[ARG0]] : !fir.ref<!fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.ptr<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[BASE_ADDR]] : !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>, !fir.type<base_t{base:i32}>) mapper(@base_t_mapper) name("obj")

// -----

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  func.func @test_polymorphic_scalar_subscript_preserves_byte_strides(%arg0: !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c20 = arith.constant 20 : index
    %b0 = omp.map.bounds lower_bound(%c2 : index) upper_bound(%c2 : index) extent(%c1 : index) stride(%c4 : index) start_idx(%c1 : index) stride_in_bytes(true)
    %b1 = omp.map.bounds lower_bound(%c1 : index) upper_bound(%c1 : index) extent(%c1 : index) stride(%c20 : index) start_idx(%c1 : index) stride_in_bytes(true)
    %0 = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>>, !fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>) map_clauses(tofrom) capture(ByRef) bounds(%b0, %b1) name("obj(3,2)") -> !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>>
    omp.target kernel_type(generic) map_entries(%0 -> %arg1 : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>>) {
      omp.terminator
    }
    return
  }
}

// CHECK-LABEL: func.func @test_polymorphic_scalar_subscript_preserves_byte_strides
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK:         %[[BOX:.*]] = fir.load %{{.*}} : !fir.ref<!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>>
// CHECK:         %[[ELEM_SIZE:.*]] = fir.box_elesize %[[BOX]] : (!fir.class<!fir.ptr<!fir.array<?x?x!fir.type<base_t{base:i32}>>>>) -> index
// CHECK:         %[[DIM1_OFFSET:.*]] = arith.muli %[[C1]], %[[C20]] : index
// CHECK:         %[[BYTE_OFFSET0:.*]] = arith.addi %{{.*}}, %[[DIM1_OFFSET]] : index
// CHECK:         %[[DIM0_OFFSET:.*]] = arith.muli %[[C2]], %[[C4]] : index
// CHECK:         %[[BYTE_OFFSET:.*]] = arith.addi %[[BYTE_OFFSET0]], %[[DIM0_OFFSET]] : index
// CHECK:         %[[BYTE_EXTENT:.*]] = arith.muli %{{.*}}, %[[ELEM_SIZE]] : index
// CHECK:         %[[UPPER:.*]] = arith.subi %{{.*}}, %{{.*}} : index
// CHECK:         omp.map.bounds lower_bound(%[[BYTE_OFFSET]] : index) upper_bound(%[[UPPER]] : index) extent(%[[BYTE_EXTENT]] : index) stride(%{{.*}} : index) start_idx(%{{.*}} : index) stride_in_bytes(true)

// -----

builtin.module attributes {omp.version = #omp.version<version = 61>} {
  omp.declare_mapper @wrapper_t_mapper : !fir.type<wrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>}> {
  ^bb0(%arg0: !fir.ref<!fir.type<wrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>}>>):
    %0 = fir.coordinate_of %arg0, item : (!fir.ref<!fir.type<wrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>
    %1 = omp.map.info var_ptr(%0 : !fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>, !fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>) map_clauses(tofrom, ref_ptee) capture(ByRef) name("w%item") -> !fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>
    omp.declare_mapper.info map_entries(%1 : !fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>)
  }
}

// CHECK-LABEL: omp.declare_mapper @wrapper_t_mapper
// CHECK:         %[[ITEM:.*]] = fir.coordinate_of %{{.*}}, item : (!fir.ref<!fir.type<wrapper_t{marker:i32,item:!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>}>>) -> !fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>
// CHECK:         %[[BASE_ADDR:.*]] = fir.box_offset %[[ITEM]] base_addr : (!fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.type<base_t{base:i32}>>>
// CHECK:         %[[TYPE_DESC:.*]] = fir.box_offset %[[ITEM]] derived_type : (!fir.ref<!fir.class<!fir.heap<!fir.type<base_t{base:i32}>>>>) -> !fir.llvm_ptr<!fir.tdesc<!fir.type<base_t{base:i32}>>>
// CHECK:         %[[TYPE_DESC_ATTACH:.*]] = omp.map.info var_ptr(%[[TYPE_DESC]] : !fir.llvm_ptr<!fir.tdesc<!fir.type<base_t{base:i32}>>>, !fir.llvm_ptr<i8>) map_clauses(attach, ref_ptee) capture(ByRef) var_ptr_ptr(%[[TYPE_DESC]] : !fir.llvm_ptr<!fir.tdesc<!fir.type<base_t{base:i32}>>>, !fir.llvm_ptr<i8>) name("w%item") -> !fir.llvm_ptr<!fir.tdesc<!fir.type<base_t{base:i32}>>>
// CHECK:         omp.declare_mapper.info map_entries({{.*}}%[[TYPE_DESC_ATTACH]]
