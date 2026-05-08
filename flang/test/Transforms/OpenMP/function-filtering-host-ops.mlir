// RUN: fir-opt --omp-function-filtering %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  // CHECK-LABEL: func.func @basic_checks
  // CHECK-SAME: (%[[ARG:.*]]: !fir.ref<i32>) -> (i32, f32)
  func.func @basic_checks(%arg: !fir.ref<i32>) -> (i32, f32) {
    // CHECK-NEXT: %[[PLACEHOLDER:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOC:.*]] = fir.convert %[[PLACEHOLDER]] : (!fir.ref<i1>) -> !fir.ref<i32>
    // CHECK-NEXT: %[[GLOBAL:.*]] = fir.address_of(@global_scalar) : !fir.ref<i32>
    %r0 = arith.constant 10 : i32
    %r1 = arith.constant 2.5 : f32

    func.call @foo() : () -> ()

    %0:2 = hlfir.declare %arg {uniq_name = "arg"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %global = fir.address_of(@global_scalar) : !fir.ref<i32>
    %1:2 = hlfir.declare %global {uniq_name = "global_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %alloc = fir.alloca i32
    %2:2 = hlfir.declare %alloc {uniq_name = "alloc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    // CHECK-NEXT: %[[ALLOC_DECL:.*]]:2 = hlfir.declare %[[ALLOC]] {uniq_name = "alloc"}
    // CHECK-NEXT: %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG]] {uniq_name = "arg"}
    // CHECK-NEXT: %[[GLOBAL_DECL:.*]]:2 = hlfir.declare %[[GLOBAL]] {uniq_name = "global_scalar"}

    // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ALLOC_DECL]]#1{{.*}})
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ARG_DECL]]#1{{.*}})
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[GLOBAL_DECL]]#1{{.*}})
    // CHECK-NEXT: %[[MAP3:.*]] = omp.map.info var_ptr(%[[ALLOC]]{{.*}})
    %m0 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m1 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m2 = omp.map.info var_ptr(%2#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m3 = omp.map.info var_ptr(%alloc : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>

    // CHECK-NEXT: omp.target has_device_addr(%[[MAP2]] -> {{.*}} : {{.*}}) map_entries(%[[MAP0]] -> {{.*}}, %[[MAP1]] -> {{.*}}, %[[MAP3]] -> {{.*}} : {{.*}})
    omp.target has_device_addr(%m2 -> %arg0 : !fir.ref<i32>) map_entries(%m0 -> %arg1, %m1 -> %arg2, %m3 -> %arg3 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
      // CHECK-NEXT: func.call
      func.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.parallel
    // CHECK-NOT: func.call
    // CHECK-NOT: omp.map.info
    omp.parallel {
      func.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.map.info
    %m4 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m5 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m6 = omp.map.info var_ptr(%2#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m7 = omp.map.info var_ptr(%alloc : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>

    // CHECK-NOT: omp.target_data
    omp.target_data map_entries(%m4, %m5, %m6, %m7 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
      // CHECK-NOT: func.call
      func.call @foo() : () -> ()
      omp.terminator
    }

    // CHECK-NOT: omp.target_enter_data
    // CHECK-NOT: omp.target_exit_data
    // CHECK-NOT: omp.target_update
    %m8 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(to) capture(ByRef) -> !fir.ref<i32>
    omp.target_enter_data map_entries(%m8 : !fir.ref<i32>)

    %m9 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(from) capture(ByRef) -> !fir.ref<i32>
    omp.target_exit_data map_entries(%m9 : !fir.ref<i32>)

    %m10 = omp.map.info var_ptr(%2#1 : !fir.ref<i32>, !fir.ref<i32>) map_clauses(to) capture(ByRef) -> !fir.ref<i32>
    omp.target_update map_entries(%m10 : !fir.ref<i32>)

    // CHECK-NOT: func.call
    func.call @foo() : () -> ()

    // CHECK:      %[[RETURN0:.*]] = fir.undefined i32
    // CHECK-NEXT: %[[RETURN1:.*]] = fir.undefined f32
    // CHECK-NEXT: return %[[RETURN0]], %[[RETURN1]]
    return %r0, %r1 : i32, f32
  }

  // CHECK-LABEL: func.func @allocatable_array
  // CHECK-SAME: (%[[ALLOCATABLE:.*]]: [[ALLOCATABLE_TYPE:.*]], %[[ARRAY:.*]]: [[ARRAY_TYPE:[^)]*]])
  func.func @allocatable_array(%allocatable: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %array: !fir.ref<!fir.array<9xi32>>) {
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[SHAPE:.*]] = fir.shape %[[ZERO]] : (i64) -> !fir.shape<1>
    // CHECK-NEXT: %[[ALLOCATABLE_DECL:.*]]:2 = hlfir.declare %[[ALLOCATABLE]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "allocatable"} : ([[ALLOCATABLE_TYPE]]) -> ([[ALLOCATABLE_TYPE]], [[ALLOCATABLE_TYPE]])
    // CHECK-NEXT: %[[ARRAY_DECL:.*]]:2 = hlfir.declare %[[ARRAY]](%[[SHAPE]]) {uniq_name = "array"} : ([[ARRAY_TYPE]], !fir.shape<1>) -> ([[ARRAY_TYPE]], [[ARRAY_TYPE]])
    // CHECK-NEXT: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[ALLOCATABLE_DECL]]#1 base_addr : ([[ALLOCATABLE_TYPE]]) -> [[VAR_PTR_PTR_TYPE:.*]]
    // CHECK-NEXT: %[[MAP_ALLOCATABLE:.*]] = omp.map.info var_ptr(%[[ALLOCATABLE_DECL]]#1 : [[ALLOCATABLE_TYPE]], f32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%[[VAR_PTR_PTR]] : [[VAR_PTR_PTR_TYPE]]) -> [[VAR_PTR_PTR_TYPE]]
    // CHECK-NEXT: %[[MAP_ARRAY:.*]] = omp.map.info var_ptr(%[[ARRAY_DECL]]#1 : [[ARRAY_TYPE]], !fir.array<9xi32>) map_clauses(tofrom) capture(ByRef) -> [[ARRAY_TYPE]]
    // CHECK-NEXT: omp.target map_entries(%[[MAP_ALLOCATABLE]] -> %{{.*}}, %[[MAP_ARRAY]] -> %{{.*}} : [[VAR_PTR_PTR_TYPE]], [[ARRAY_TYPE]])
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index

    %0:2 = hlfir.declare %allocatable {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "allocatable"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
    %1 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%c8 : index) extent(%c9 : index) stride(%c1 : index) start_idx(%c1 : index)
    %2 = fir.box_offset %0#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
    %m0 = omp.map.info var_ptr(%0#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, f32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%2 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) bounds(%1) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>

    %3 = fir.shape %c9 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %array(%3) {uniq_name = "array"} : (!fir.ref<!fir.array<9xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<9xi32>>, !fir.ref<!fir.array<9xi32>>)
    %5 = omp.map.bounds lower_bound(%c0 : index) upper_bound(%c8 : index) extent(%c9 : index) stride(%c1 : index) start_idx(%c1 : index)
    %6 = omp.map.info var_ptr(%4#1 : !fir.ref<!fir.array<9xi32>>, !fir.array<9xi32>) map_clauses(tofrom) capture(ByRef) bounds(%5) -> !fir.ref<!fir.array<9xi32>>

    omp.target map_entries(%m0 -> %arg0, %6 -> %arg1 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>, !fir.ref<!fir.array<9xi32>>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @character
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^)]*]])
  func.func @character(%x: !fir.ref<!fir.char<1>>) {
    // CHECK-NEXT: %[[ZERO]] = arith.constant 0 : i64
    %0 = fir.dummy_scope : !fir.dscope
    %c1 = arith.constant 1 : index
    // CHECK-NEXT: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] typeparams %[[ZERO]] {uniq_name = "x"} : ([[X_TYPE]], i64) -> ([[X_TYPE]], [[X_TYPE]])
    %3:2 = hlfir.declare %x typeparams %c1 dummy_scope %0 {uniq_name = "x"} : (!fir.ref<!fir.char<1>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
    // CHECK-NEXT: %[[MAP:.*]] = omp.map.info var_ptr(%[[X_DECL]]#1 : [[X_TYPE]], !fir.char<1>) map_clauses(tofrom) capture(ByRef) -> [[X_TYPE]]
    %map = omp.map.info var_ptr(%3#1 : !fir.ref<!fir.char<1>>, !fir.char<1>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.char<1>>
    // CHECK-NEXT: omp.target map_entries(%[[MAP]] -> %{{.*}})
    omp.target map_entries(%map -> %arg0 : !fir.ref<!fir.char<1>>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @assumed_rank
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^)]*]])
  func.func @assumed_rank(%x: !fir.box<!fir.array<*:f32>>) {
    // CHECK-NEXT: %[[PLACEHOLDER:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOCA:.*]] = fir.convert %[[PLACEHOLDER]] : (!fir.ref<i1>) -> !fir.ref<[[X_TYPE]]>
    %0 = fir.alloca !fir.box<!fir.array<*:f32>>
    %1 = fir.dummy_scope : !fir.dscope
    %2:2 = hlfir.declare %x dummy_scope %1 {uniq_name = "x"} : (!fir.box<!fir.array<*:f32>>, !fir.dscope) -> (!fir.box<!fir.array<*:f32>>, !fir.box<!fir.array<*:f32>>)
    %3 = fir.box_addr %2#1 : (!fir.box<!fir.array<*:f32>>) -> !fir.ref<!fir.array<*:f32>>
    fir.store %2#1 to %0 : !fir.ref<!fir.box<!fir.array<*:f32>>>
    // CHECK-NEXT: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[ALLOCA]] base_addr : (!fir.ref<[[X_TYPE]]>) -> [[VAR_PTR_PTR_TYPE:.*]]
    %4 = fir.box_offset %0 base_addr : (!fir.ref<!fir.box<!fir.array<*:f32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<*:f32>>>
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<[[X_TYPE]]>, !fir.array<*:f32>) {{.*}} var_ptr_ptr(%[[VAR_PTR_PTR]] : [[VAR_PTR_PTR_TYPE]]) -> [[VAR_PTR_PTR_TYPE]]
    %5 = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.array<*:f32>>>, !fir.array<*:f32>) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%4 : !fir.llvm_ptr<!fir.ref<!fir.array<*:f32>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<*:f32>>>
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ALLOCA]] : !fir.ref<[[X_TYPE]]>, !fir.box<!fir.array<*:f32>>) {{.*}} members(%[[MAP0]] : [0] : [[VAR_PTR_PTR_TYPE]]) -> !fir.ref<!fir.array<*:f32>>
    %6 = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.array<*:f32>>>, !fir.box<!fir.array<*:f32>>) map_clauses(to) capture(ByRef) members(%5 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<*:f32>>>) -> !fir.ref<!fir.array<*:f32>>
    // CHECK-NEXT: omp.target map_entries(%[[MAP1]] -> %{{.*}}, %[[MAP0]] -> {{.*}})
    omp.target map_entries(%6 -> %arg1, %5 -> %arg2 : !fir.ref<!fir.array<*:f32>>, !fir.llvm_ptr<!fir.ref<!fir.array<*:f32>>>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @box_ptr
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^)]*]])
  func.func @box_ptr(%x: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) {
    // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0 : i64
    // CHECK-NEXT: %[[SHAPE:.*]] = fir.shape_shift %[[ZERO]], %[[ZERO]] : (i64, i64) -> !fir.shapeshift<1>
    // CHECK-NEXT: %[[PLACEHOLDER_X:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOCA_X:.*]] = fir.convert %[[PLACEHOLDER_X]] : (!fir.ref<i1>) -> [[X_TYPE]]
    %0 = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
    %1 = fir.dummy_scope : !fir.dscope
    %2:2 = hlfir.declare %x dummy_scope %1 {fortran_attrs = #fir.var_attrs<contiguous, pointer>, uniq_name = "x"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
    %3 = fir.load %2#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    fir.store %3 to %0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

    // CHECK-NEXT: %[[PLACEHOLDER_Y:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOCA_Y:.*]] = fir.convert %[[PLACEHOLDER_Y]] : (!fir.ref<i1>) -> [[Y_TYPE:.*]]
    %c0 = arith.constant 0 : index
    %4:3 = fir.box_dims %3, %c0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c1 = arith.constant 1 : index
    %c0_0 = arith.constant 0 : index
    %5:3 = fir.box_dims %3, %c0_0 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %c0_1 = arith.constant 0 : index
    %6 = arith.subi %5#1, %c1 : index
    %7 = omp.map.bounds lower_bound(%c0_1 : index) upper_bound(%6 : index) extent(%5#1 : index) stride(%5#2 : index) start_idx(%4#0 : index) {stride_in_bytes = true}
    %8 = fir.box_addr %3 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
    %c0_2 = arith.constant 0 : index
    %9:3 = fir.box_dims %3, %c0_2 : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
    %10 = fir.shape_shift %9#0, %9#1 : (index, index) -> !fir.shapeshift<1>
    
    // CHECK-NEXT: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ALLOCA_Y]](%[[SHAPE]]) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "y"} : ([[Y_TYPE]], !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, [[Y_TYPE]])
    %11:2 = hlfir.declare %8(%10) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "y"} : (!fir.ptr<!fir.array<?xi32>>, !fir.shapeshift<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.ptr<!fir.array<?xi32>>)
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %12:3 = fir.box_dims %11#0, %c0_4 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
    %c0_5 = arith.constant 0 : index
    %13 = arith.subi %12#1, %c1_3 : index
    %14 = omp.map.bounds lower_bound(%c0_5 : index) upper_bound(%13 : index) extent(%12#1 : index) stride(%12#2 : index) start_idx(%9#0 : index) {stride_in_bytes = true}
    
    // CHECK-NEXT: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[ALLOCA_X]] base_addr : ([[X_TYPE]]) -> [[VAR_PTR_PTR_TYPE:.*]]
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[Y_DECL]]#1 : [[Y_TYPE]], i32) {{.*}} -> [[Y_TYPE]]
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[ALLOCA_X]] : [[X_TYPE]], i32) {{.*}} var_ptr_ptr(%[[VAR_PTR_PTR]] : [[VAR_PTR_PTR_TYPE]]) -> [[VAR_PTR_PTR_TYPE]]
    // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[ALLOCA_X]] : [[X_TYPE]], !fir.box<!fir.ptr<!fir.array<?xi32>>>) {{.*}} members(%[[MAP1]] : [0] : [[VAR_PTR_PTR_TYPE]]) -> [[X_TYPE]]
    %15 = omp.map.info var_ptr(%11#1 : !fir.ptr<!fir.array<?xi32>>, i32) map_clauses(tofrom) capture(ByRef) bounds(%14) -> !fir.ptr<!fir.array<?xi32>>
    %16 = fir.box_offset %0 base_addr : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %17 = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, i32) map_clauses(implicit, to) capture(ByRef) var_ptr_ptr(%16 : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) bounds(%7) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>
    %18 = omp.map.info var_ptr(%0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.box<!fir.ptr<!fir.array<?xi32>>>) map_clauses(implicit, to) capture(ByRef) members(%17 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
    
    // CHECK-NEXT: omp.target map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP2]] -> %{{.*}}, %[[MAP1]] -> {{.*}} : [[Y_TYPE]], [[X_TYPE]], [[VAR_PTR_PTR_TYPE]])
    omp.target map_entries(%15 -> %arg1, %18 -> %arg2, %17 -> %arg3 : !fir.ptr<!fir.array<?xi32>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xi32>>>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @target_data
  // CHECK-SAME: (%[[MAPPED:.*]]: [[MAPPED_TYPE:[^)]*]], %[[USEDEVADDR:.*]]: [[USEDEVADDR_TYPE:[^)]*]], %[[USEDEVPTR:.*]]: [[USEDEVPTR_TYPE:[^)]*]])
  func.func @target_data(%mapped: !fir.ref<i32>, %usedevaddr: !fir.ref<i32>, %usedevptr: !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
    // CHECK-NEXT: %[[MAPPED_DECL:.*]]:2 = hlfir.declare %[[MAPPED]] {uniq_name = "mapped"} : ([[MAPPED_TYPE]]) -> ([[MAPPED_TYPE]], [[MAPPED_TYPE]])
    // CHECK-NEXT: %[[USEDEVADDR_DECL:.*]]:2 = hlfir.declare %[[USEDEVADDR]] {uniq_name = "usedevaddr"} : ([[USEDEVADDR_TYPE]]) -> ([[USEDEVADDR_TYPE]], [[USEDEVADDR_TYPE]])
    // CHECK-NEXT: %[[USEDEVPTR_DECL:.*]]:2 = hlfir.declare %[[USEDEVPTR]] {uniq_name = "usedevptr"} : ([[USEDEVPTR_TYPE]]) -> ([[USEDEVPTR_TYPE]], [[USEDEVPTR_TYPE]])
    %0:2 = hlfir.declare %mapped {uniq_name = "mapped"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %1:2 = hlfir.declare %usedevaddr {uniq_name = "usedevaddr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2:2 = hlfir.declare %usedevptr {uniq_name = "usedevptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)

    // CHECK-NEXT: %[[MAPPED_MAP:.*]] = omp.map.info var_ptr(%[[MAPPED_DECL]]#1 : [[MAPPED_TYPE]], i32) map_clauses(tofrom) capture(ByRef) -> [[MAPPED_TYPE]]
    // CHECK-NEXT: %[[USEDEVADDR_MAP:.*]] = omp.map.info var_ptr(%[[USEDEVADDR_DECL]]#1 : [[USEDEVADDR_TYPE]], i32) map_clauses(tofrom) capture(ByRef) -> [[USEDEVADDR_TYPE]]
    // CHECK-NEXT: %[[USEDEVPTR_MAP:.*]] = omp.map.info var_ptr(%[[USEDEVPTR_DECL]]#1 : [[USEDEVPTR_TYPE]], !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) map_clauses(tofrom) capture(ByRef) -> [[USEDEVPTR_TYPE]]
    %m0 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %m1 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(return_param) capture(ByRef) -> !fir.ref<i32>
    %m2 = omp.map.info var_ptr(%2#1 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) map_clauses(return_param) capture(ByRef) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
    
    // CHECK-NOT: omp.target_data
    omp.target_data map_entries(%m0 : !fir.ref<i32>) use_device_addr(%m1 -> %arg0 : !fir.ref<i32>) use_device_ptr(%m2 -> %arg1 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
      %3:2 = hlfir.declare %arg0 {uniq_name = "usedevaddr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %4:2 = hlfir.declare %arg1 {uniq_name = "usedevptr"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) -> (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>)
      %m3 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      %m4 = omp.map.info var_ptr(%3#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      %m5 = omp.map.info var_ptr(%4#1 : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>, !fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>) map_clauses(tofrom) capture(ByRef) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>

      // CHECK-NOT: func.call
      func.call @foo() : () -> ()

      // CHECK-NEXT: omp.target map_entries(%[[MAPPED_MAP]] -> %{{.*}}, %[[USEDEVADDR_MAP]] -> %{{.*}}, %[[USEDEVPTR_MAP]] -> %{{.*}} : {{.*}})
      omp.target map_entries(%m3 -> %arg2, %m4 -> %arg3, %m5 -> %arg4 : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>) {
        omp.terminator
      }

      // CHECK-NOT: func.call
      func.call @foo() : () -> ()

      omp.terminator
    }

    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @map_info_members
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^)]*]])
  func.func @map_info_members(%x: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    // CHECK-NEXT: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : ([[X_TYPE]]) -> ([[X_TYPE]], [[X_TYPE]])
    %23:2 = hlfir.declare %x {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "x"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
    %63 = fir.load %23#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %64:3 = fir.box_dims %63, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %65:3 = fir.box_dims %63, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %66 = arith.subi %c1, %64#0 : index
    %67 = arith.subi %c9, %64#0 : index
    %68 = fir.load %23#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    %69:3 = fir.box_dims %68, %c0 : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
    %70 = omp.map.bounds lower_bound(%66 : index) upper_bound(%67 : index) extent(%69#1 : index) stride(%65#2 : index) start_idx(%64#0 : index) {stride_in_bytes = true}
    // CHECK-NEXT: %[[VAR_PTR_PTR:.*]] = fir.box_offset %[[X_DECL]]#1 base_addr : ([[X_TYPE]]) -> [[VAR_PTR_PTR_TYPE:.*]]
    %71 = fir.box_offset %23#1 base_addr : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[X_DECL]]#1 : [[X_TYPE]], f32) {{.*}} var_ptr_ptr(%[[VAR_PTR_PTR]] : [[VAR_PTR_PTR_TYPE]]) -> [[VAR_PTR_PTR_TYPE]]
    %72 = omp.map.info var_ptr(%23#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, f32) map_clauses(tofrom) capture(ByRef) var_ptr_ptr(%71 : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) bounds(%70) -> !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[X_DECL]]#1 : [[X_TYPE]], !fir.box<!fir.heap<!fir.array<?xf32>>>) {{.*}} members(%[[MAP0]] : [0] : [[VAR_PTR_PTR_TYPE]]) -> [[X_TYPE]]
    %73 = omp.map.info var_ptr(%23#1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(to) capture(ByRef) members(%72 : [0] : !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    // CHECK-NEXT: omp.target map_entries(%[[MAP1]] -> {{.*}}, %[[MAP0]] -> %{{.*}} : [[X_TYPE]], [[VAR_PTR_PTR_TYPE]])
    omp.target map_entries(%73 -> %arg0, %72 -> %arg1 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.llvm_ptr<!fir.ref<!fir.array<?xf32>>>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @control_flow
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^,]*]], %[[COND:.*]]: [[COND_TYPE:[^)]*]])
  func.func @control_flow(%x: !fir.ref<i32>, %cond: !fir.ref<!fir.logical<4>>) {
    // CHECK-NEXT: %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "x"} : ([[X_TYPE]]) -> ([[X_TYPE]], [[X_TYPE]])
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[X_DECL]]#1 : [[X_TYPE]], i32) {{.*}} -> [[X_TYPE]]
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[X_DECL]]#1 : [[X_TYPE]], i32) {{.*}} -> [[X_TYPE]]
    %x_decl:2 = hlfir.declare %x {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %cond_decl:2 = hlfir.declare %cond {uniq_name = "cond"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
    %0 = fir.load %cond_decl#0 : !fir.ref<!fir.logical<4>>
    %1 = fir.convert %0 : (!fir.logical<4>) -> i1
    cf.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    fir.call @foo() : () -> ()
    %m0 = omp.map.info var_ptr(%x_decl#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK-NEXT: omp.target map_entries(%[[MAP0]] -> {{.*}} : [[X_TYPE]])
    omp.target map_entries(%m0 -> %arg2 : !fir.ref<i32>) {
      omp.terminator
    }
    fir.call @foo() : () -> ()
    cf.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    // CHECK-NOT: fir.call
    // CHECK-NOT: omp.map.info
    // CHECK-NOT: omp.target_data
    fir.call @foo() : () -> ()
    %m1 = omp.map.info var_ptr(%x_decl#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    omp.target_data map_entries(%m1 : !fir.ref<i32>) {
      fir.call @foo() : () -> ()
      %8 = fir.load %cond_decl#0 : !fir.ref<!fir.logical<4>>
      %9 = fir.convert %8 : (!fir.logical<4>) -> i1
      cf.cond_br %9, ^bb1, ^bb2
    ^bb1:  // pred: ^bb0
      fir.call @foo() : () -> ()
      %m2 = omp.map.info var_ptr(%x_decl#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      // CHECK: omp.target map_entries(%[[MAP1]] -> {{.*}} : [[X_TYPE]])
      omp.target map_entries(%m2 -> %arg2 : !fir.ref<i32>) {
        omp.terminator
      }
      // CHECK-NOT: fir.call
      // CHECK-NOT: cf.br
      fir.call @foo() : () -> ()
      cf.br ^bb2
    ^bb2:  // 2 preds: ^bb0, ^bb1
      fir.call @foo() : () -> ()
      omp.terminator
    }
    fir.call @foo() : () -> ()

    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @block_args
  // CHECK-SAME: (%[[X:.*]]: [[X_TYPE:[^)]*]])
  func.func @block_args(%x: !fir.ref<i32>) {
    // CHECK-NEXT: %[[PLACEHOLDER0:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOCA0:.*]] = fir.convert %[[PLACEHOLDER0]] : (!fir.ref<i1>) -> !fir.ref<i32>
    // CHECK-NEXT: %[[PLACEHOLDER1:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[ALLOCA1:.*]] = fir.convert %[[PLACEHOLDER1]] : (!fir.ref<i1>) -> !fir.ref<i32>
    // CHECK-NEXT: %[[X_DECL0:.*]]:2 = hlfir.declare %[[ALLOCA0]] {uniq_name = "x"} : ([[X_TYPE]]) -> ([[X_TYPE]], [[X_TYPE]])
    // CHECK-NEXT: %[[X_DECL1:.*]]:2 = hlfir.declare %[[ALLOCA1]] {uniq_name = "x"} : ([[X_TYPE]]) -> ([[X_TYPE]], [[X_TYPE]])
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[X_DECL0]]#1 : [[X_TYPE]], i32) {{.*}} -> [[X_TYPE]]
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[X_DECL1]]#1 : [[X_TYPE]], i32) {{.*}} -> [[X_TYPE]]
    %x_decl:2 = hlfir.declare %x {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    omp.parallel private(@privatizer %x_decl#0 -> %arg0 : !fir.ref<i32>) {
      %0:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %m0 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      // CHECK-NEXT: omp.target map_entries(%[[MAP0]] -> {{.*}} : [[X_TYPE]])
      omp.target map_entries(%m0 -> %arg2 : !fir.ref<i32>) {
        omp.terminator
      }
      omp.terminator
    }

    // CHECK-NOT: omp.parallel
    // CHECK-NOT: hlfir.declare
    // CHECK-NOT: omp.map.info
    // CHECK-NOT: omp.target_data
    omp.parallel private(@privatizer %x_decl#0 -> %arg0 : !fir.ref<i32>) {
      %1:2 = hlfir.declare %arg0 {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %m1 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      omp.target_data map_entries(%m1 : !fir.ref<i32>) {
        omp.parallel private(@privatizer %1#0 -> %arg1 : !fir.ref<i32>) {
          %2:2 = hlfir.declare %arg1 {uniq_name = "x"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
          %m2 = omp.map.info var_ptr(%2#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
          // CHECK: omp.target map_entries(%[[MAP1]] -> {{.*}} : [[X_TYPE]])
          omp.target map_entries(%m2 -> %arg2 : !fir.ref<i32>) {
            omp.terminator
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }

    return
  }

  // CHECK-LABEL: func.func @reuse_tests()
  func.func @reuse_tests() {
    // CHECK-NEXT: %[[PLACEHOLDER:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[THREAD_LIMIT:.*]] = fir.convert %[[PLACEHOLDER]] : (!fir.ref<i1>) -> i32
    // CHECK-NEXT: %[[CONST:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[GLOBAL:.*]] = fir.address_of(@global_scalar) : !fir.ref<i32>
    %global = fir.address_of(@global_scalar) : !fir.ref<i32>
    // CHECK-NEXT: %[[GLOBAL_DECL0:.*]]:2 = hlfir.declare %[[GLOBAL]] {uniq_name = "global_scalar"}
    // CHECK-NEXT: %[[GLOBAL_DECL1:.*]]:2 = hlfir.declare %[[GLOBAL]] {uniq_name = "global_scalar"}
    // CHECK-NEXT: %[[GLOBAL_DECL2:.*]]:2 = hlfir.declare %[[GLOBAL]] {uniq_name = "global_scalar"}
    // CHECK-NEXT: %[[MAP0:.*]] = omp.map.info var_ptr(%[[GLOBAL_DECL0]]#1 : !fir.ref<i32>, i32)
    // CHECK-NEXT: %[[MAP1:.*]] = omp.map.info var_ptr(%[[GLOBAL_DECL1]]#1 : !fir.ref<i32>, i32)
    // CHECK-NEXT: %[[MAP2:.*]] = omp.map.info var_ptr(%[[GLOBAL_DECL2]]#1 : !fir.ref<i32>, i32)
    %0:2 = hlfir.declare %global {uniq_name = "global_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %m0 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK-NOT: omp.target_data
    omp.target_data map_entries(%m0 : !fir.ref<i32>) {
      %1:2 = hlfir.declare %global {uniq_name = "global_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
      %m1 = omp.map.info var_ptr(%0#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      %m2 = omp.map.info var_ptr(%1#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
      // CHECK-NEXT: omp.target map_entries(%[[MAP0]] -> %{{.*}}, %[[MAP1]] -> {{.*}} : !fir.ref<i32>, !fir.ref<i32>)
      omp.target map_entries(%m1 -> %arg0, %m2 -> %arg1 : !fir.ref<i32>, !fir.ref<i32>) {
        omp.terminator
      }
      omp.terminator
    }
    // CHECK-NOT: fir.load
    // CHECK-NOT: hlfir.declare
    // CHECK-NOT: omp.map.info
    %2 = fir.load %global : !fir.ref<i32>
    %3:2 = hlfir.declare %global {uniq_name = "global_scalar"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %m3 = omp.map.info var_ptr(%3#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK: omp.target thread_limit(%[[THREAD_LIMIT]] : i32) map_entries(%[[MAP2]] -> %{{.*}} : !fir.ref<i32>)
    omp.target thread_limit(%2 : i32) map_entries(%m3 -> %arg0 : !fir.ref<i32>) {
      omp.terminator
    }
    // CHECK: omp.target thread_limit(%[[CONST]] : i32)
    %c1 = arith.constant 1 : i32
    omp.target thread_limit(%c1 : i32) {
      omp.terminator
    }
    // CHECK: omp.target thread_limit(%[[CONST]] : i32)
    omp.target thread_limit(%c1 : i32) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @all_non_map_clauses
  // CHECK-SAME: (%[[REF:.*]]: !fir.ref<i32>, %[[INT:.*]]: i32, %[[BOOL:.*]]: i1)
  func.func @all_non_map_clauses(%ref: !fir.ref<i32>, %int: i32, %bool: i1) {
    %m0 = omp.map.info var_ptr(%ref : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK-NOT: omp.target_data
    omp.target_data device(%int : i32) if(%bool) map_entries(%m0 : !fir.ref<i32>) {
      omp.terminator
    }
    // CHECK-NEXT: omp.target allocate({{[^)]*}}) thread_limit({{[^)]*}}) in_reduction({{[^)]*}}) private({{[^)]*}}) {
    omp.target allocate(%ref : !fir.ref<i32> -> %ref : !fir.ref<i32>)
               depend(taskdependin -> %ref : !fir.ref<i32>)
               device(%int : i32) if(%bool) thread_limit(%int : i32)
               in_reduction(@reduction %ref -> %arg0 : !fir.ref<i32>)
               private(@privatizer %ref -> %arg1 : !fir.ref<i32>) {
      omp.terminator
    }
    // CHECK-NOT: omp.target_enter_data
    omp.target_enter_data depend(taskdependin -> %ref : !fir.ref<i32>)
                          device(%int : i32) if(%bool)
    // CHECK-NOT: omp.target_exit_data
    omp.target_exit_data depend(taskdependin -> %ref : !fir.ref<i32>)
                         device(%int : i32) if(%bool)
    // CHECK-NOT: omp.target_update
    omp.target_update depend(taskdependin -> %ref : !fir.ref<i32>)
                      device(%int : i32) if(%bool)

    // CHECK: return
    return
  }

  // CHECK-LABEL: func.func @assumed_length
  // CHECK-SAME: (%[[ARG:.*]]: !fir.boxchar<1>)
  func.func @assumed_length(%arg: !fir.boxchar<1>) {
    // CHECK-NEXT: %[[PLACEHOLDER:.*]] = fir.alloca !fir.char<1>
    // CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[EMBOXCHAR:.*]] = fir.emboxchar %[[PLACEHOLDER]], %[[ONE]] : (!fir.ref<!fir.char<1>>, i32) -> !fir.boxchar<1>
    // CHECK-NEXT: omp.target private(@boxchar_firstprivatizer %[[EMBOXCHAR]] -> %{{.*}} [map_idx=0] : !fir.boxchar<1>)
    %0 = fir.alloca !fir.boxchar<1>
    %1 = fir.dummy_scope : !fir.dscope
    %2:2 = fir.unboxchar %arg : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
    %3:2 = hlfir.declare %2#0 typeparams %2#1 dummy_scope %1 {uniq_name = "arg"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    omp.target private(@boxchar_firstprivatizer %3#0 -> %arg3 [map_idx=0] : !fir.boxchar<1>) {
      omp.terminator
    }
    return
  }

  // CHECK-LABEL: func.func @hlfir_storage
  func.func @hlfir_storage(%i : index) {
    // CHECK-NEXT: %[[PLACEHOLDER:.*]] = fir.alloca i1
    // CHECK-NEXT: %[[VAR:.*]] = fir.convert %[[PLACEHOLDER]] : (!fir.ref<i1>) -> !fir.ref<i32>
    // CHECK-NEXT: %[[GLOBAL:.*]] = fir.address_of(@block_) : !fir.ref<!fir.array<8xi8>>
    %0 = fir.address_of(@block_) : !fir.ref<!fir.array<8xi8>>
    %1 = fir.convert %0 : (!fir.ref<!fir.array<8xi8>>) -> !fir.ref<!fir.array<?xi8>>
    %2 = fir.coordinate_of %1, %i : (!fir.ref<!fir.array<?xi8>>, index) -> !fir.ref<i8>
    %3 = fir.convert %2 : (!fir.ref<i8>) -> !fir.ref<i32>
    // CHECK-NEXT: %[[DECL:.*]]:2 = hlfir.declare %[[VAR]] storage(%[[GLOBAL]][0])
    %4:2 = hlfir.declare %3 storage (%0[0]) {uniq_name = "a"} : (!fir.ref<i32>, !fir.ref<!fir.array<8xi8>>) -> (!fir.ref<i32>, !fir.ref<i32>)
    // CHECK-NEXT: %[[MAP:.*]] = omp.map.info var_ptr(%[[DECL]]#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    %map = omp.map.info var_ptr(%4#1 : !fir.ref<i32>, i32) map_clauses(tofrom) capture(ByRef) -> !fir.ref<i32>
    // CHECK-NEXT: omp.target map_entries(%[[MAP]] -> %{{.*}} : !fir.ref<i32>)
    omp.target map_entries(%map -> %arg0 : !fir.ref<i32>) {
      omp.terminator
    }
    return
  }

  func.func private @foo() -> () attributes {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (enter)>}
  fir.global internal @global_scalar constant : i32 {
    %0 = arith.constant 10 : i32
    fir.has_value %0 : i32
  }
  omp.private {type = firstprivate} @privatizer : i32 copy {
  ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    %0 = fir.load %arg0 : !fir.ref<i32>
    hlfir.assign %0 to %arg1 : i32, !fir.ref<i32>
    omp.yield(%arg1 : !fir.ref<i32>)
  }
  omp.declare_reduction @reduction : i32
  init {
  ^bb0(%arg: i32):
    %0 = arith.constant 0 : i32
    omp.yield (%0 : i32)
  }
  combiner {
  ^bb1(%arg0: i32, %arg1: i32):
    %1 = arith.addi %arg0, %arg1 : i32
    omp.yield (%1 : i32)
  }
}
