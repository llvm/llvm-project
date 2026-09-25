// RUN: inter-opt %s --inter-import-llvm --lift-cf-to-scf \
// RUN:   --inter-verify-structured --inter-convert-llvm-to-xw | FileCheck %s
// RUN: inter-opt %s '--inter-import-llvm=simd-width=32' --lift-cf-to-scf \
// RUN:   --inter-verify-structured --inter-convert-llvm-to-xw | \
// RUN:   FileCheck %s --check-prefix=WIDTH32

module {
  llvm.module_flags [#llvm.mlir.module_flag<warning, "Debug Info Version",
                                             3 : i32>]

  llvm.func spir_kernelcc @vector_add(%out: !llvm.ptr<1>,
                                      %in: !llvm.ptr<1>) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %input_ptr = llvm.getelementptr %in[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %output_ptr = llvm.getelementptr %out[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %value = llvm.load %input_ptr : !llvm.ptr<1> -> i32
    %sum = llvm.add %value, %one : i32
    llvm.store %sum, %output_ptr : i32, !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_kernelcc @negative_gep(%base: !llvm.ptr<1>, %index: i32) {
    %pointer = llvm.getelementptr %base[%index]
        : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
    llvm.return
  }
  llvm.func spir_kernelcc @local_gep(%base: !llvm.ptr<3>, %index: i64) {
    %pointer = llvm.getelementptr %base[%index]
        : (!llvm.ptr<3>, i64) -> !llvm.ptr<3>, i32
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64

  llvm.mlir.global internal @scratch() {addr_space = 3 : i32} : i32
  llvm.mlir.global internal @scratch_wide() {
    addr_space = 3 : i32, alignment = 16 : i64} : i64

  llvm.func spir_kernelcc @spaces(%p0: !llvm.ptr, %p1: !llvm.ptr<1>,
                                  %p2: !llvm.ptr<2>, %p3: !llvm.ptr<3>,
                                  %p4: !llvm.ptr<4>, %condition: i1) {
    %local = llvm.mlir.addressof @scratch : !llvm.ptr<3>
    %local_wide = llvm.mlir.addressof @scratch_wide : !llvm.ptr<3>
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %sum = llvm.add %zero, %one overflow<nsw, nuw> : i32
    %wide = llvm.sext %sum : i32 to i64
    %narrow = arith.trunci %sum overflow<nuw> : i32 to i1
    %cmp = llvm.icmp "eq" %sum, %one : i32
    %inverted = arith.xori %cmp, %condition : i1
    %selected = llvm.select %cmp, %sum, %one : i1, i32
    llvm.cond_br %condition, ^then, ^else
  ^then:
    llvm.br ^merge(%selected : i32)
  ^else:
    llvm.br ^merge(%zero : i32)
  ^merge(%result: i32):
    llvm.store %result, %local : i32, !llvm.ptr<3>
    llvm.store %wide, %local_wide : i64, !llvm.ptr<3>
    llvm.return
  }

  llvm.func spir_kernelcc @queries_and_math(%pointer: !llvm.ptr<1>,
                                             %lhs: f32, %rhs: f32) {
    %axis = llvm.mlir.constant(1 : i32) : i32
    %lane = llvm.call spir_funccc @_Z22get_sub_group_local_id() : () -> i32
    %subgroup = llvm.call spir_funccc @_Z16get_sub_group_id() : () -> i32
    %lid = llvm.call spir_funccc @_Z12get_local_idm(%axis) : (i32) -> i64
    %group = llvm.call spir_funccc @_Z12get_group_idm(%axis) : (i32) -> i64
    %global_size = llvm.call spir_funccc @_Z15get_global_sizem(%axis) : (i32) -> i64
    %local_size = llvm.call spir_funccc @_Z14get_local_sizem(%axis) : (i32) -> i64
    %groups = llvm.call spir_funccc @_Z14get_num_groupsm(%axis) : (i32) -> i64
    %grid = llvm.call spir_funccc @__builtin_IB_get_global_size(%axis) : (i32) -> i64
    %block = llvm.call spir_funccc @__builtin_IB_get_local_size(%axis) : (i32) -> i64
    %zero = llvm.mlir.constant(0 : i64) : i64
    %null = llvm.inttoptr %zero : i64 to !llvm.ptr<1>
    %eq = llvm.icmp "eq" %pointer, %null : !llvm.ptr<1>
    %ne = llvm.icmp "ne" %pointer, %null : !llvm.ptr<1>
    %sum = llvm.fadd %lhs, %rhs : f32
    %difference = llvm.fsub %sum, %rhs : f32
    %product = llvm.fmul %difference, %rhs : f32
    %old = llvm.atomicrmw add %pointer, %lane monotonic : !llvm.ptr<1>, i32
    %builtin_old = llvm.call spir_funccc @_Z10atomic_addPU3AS1Vjj(%pointer, %lane)
        : (!llvm.ptr<1>, i32) -> i32
    llvm.return
  }

  llvm.func spir_funccc @_Z22get_sub_group_local_id() -> i32
  llvm.func spir_funccc @_Z16get_sub_group_id() -> i32
  llvm.func spir_funccc @_Z12get_local_idm(i32) -> i64
  llvm.func spir_funccc @_Z12get_group_idm(i32) -> i64
  llvm.func spir_funccc @_Z15get_global_sizem(i32) -> i64
  llvm.func spir_funccc @_Z14get_local_sizem(i32) -> i64
  llvm.func spir_funccc @_Z14get_num_groupsm(i32) -> i64
  llvm.func spir_funccc @__builtin_IB_get_global_size(i32) -> i64
  llvm.func spir_funccc @__builtin_IB_get_local_size(i32) -> i64
  llvm.func spir_funccc @_Z10atomic_addPU3AS1Vjj(!llvm.ptr<1>, i32) -> i32

  llvm.func spir_kernelcc @poison_freeze(%pointer: !llvm.ptr<1>) {
    %llvm_poison = llvm.mlir.poison : i32
    %frozen_poison = llvm.freeze %llvm_poison {boundary.note = "kept"} : i32
    %lifted_poison = ub.poison : !llvm.ptr<1>
    %frozen_pointer = llvm.freeze %lifted_poison : !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_kernelcc @mixed(%pointer: !llvm.ptr<1>, %integer: i64,
                                 %floating: f32) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %integer_cmp = llvm.icmp "ult" %gid, %integer : i64
    %integer_reverse = llvm.icmp "ugt" %integer, %gid : i64
    %element = llvm.getelementptr %pointer[%gid]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    %pointer_cmp = llvm.icmp "eq" %element, %pointer : !llvm.ptr<1>
    %loaded = llvm.load %element : !llvm.ptr<1> -> f32
    %float_cmp = llvm.fcmp "olt" %loaded, %floating : f32
    %selected_integer = llvm.select %integer_cmp, %gid, %integer : i1, i64
    %selected_pointer = llvm.select %pointer_cmp, %element, %pointer
        : i1, !llvm.ptr<1>
    %selected_float = llvm.select %float_cmp, %floating, %loaded : i1, f32
    llvm.return
  }

  llvm.func spir_kernelcc @divergent_if(%out: !llvm.ptr<1>, %limit: i64) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %active = llvm.icmp "ult" %gid, %limit : i64
    llvm.cond_br %active, ^then, ^else
  ^then:
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^merge(%one : i32)
  ^else:
    %zero = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^merge(%zero : i32)
  ^merge(%value: i32):
    llvm.store %value, %out : i32, !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_kernelcc @mixed_divergent_if(%out: !llvm.ptr<1>,
                                               %limit: i64) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %gid32 = llvm.trunc %gid : i64 to i32
    %active = llvm.icmp "ult" %gid, %limit : i64
    llvm.cond_br %active, ^then, ^else
  ^then:
    llvm.br ^merge(%gid32 : i32)
  ^else:
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^merge(%one : i32)
  ^merge(%value: i32):
    llvm.store %value, %out : i32, !llvm.ptr<1>
    llvm.return
  }

  llvm.func spir_kernelcc @one_sided_divergent(%out: !llvm.ptr<1>,
                                                %limit: i64) {
    %axis = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%axis) : (i32) -> i64
    %active = llvm.icmp "ult" %gid, %limit : i64
    llvm.cond_br %active, ^then, ^merge
  ^then:
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %out : i32, !llvm.ptr<1>
    llvm.br ^merge
  ^merge:
    llvm.return
  }
}

// CHECK-NOT: llvm.module_flags
// CHECK-LABEL: func.func @vector_add(%{{.*}}: !xw.ptr<#xw.global>
// CHECK: xw.global_id 0
// CHECK: xw.ptradd
// CHECK: xw.load
// CHECK: xw.binary addi
// CHECK: xw.store
// CHECK-NOT: llvm.
// CHECK-NOT: {{(^|[^s])cf\.}}

// CHECK-LABEL: func.func @negative_gep
// CHECK: xw.cast intconvert {{.*}}policy {extension = #xw.cast_extension<sign>}
// CHECK: xw.ptradd

// CHECK-LABEL: func.func @local_gep
// CHECK: xw.cast intconvert {{.*}} : i64 -> i32
// CHECK: xw.ptradd {{.*}} : !xw.ptr<#xw.local>, i32 -> !xw.ptr<#xw.local>

// CHECK-LABEL: func.func @spaces(
// CHECK-SAME: !xw.ptr<#xw.private>
// CHECK-SAME: !xw.ptr<#xw.global>
// CHECK-SAME: !xw.ptr<#xw.constant>
// CHECK-SAME: !xw.ptr<#xw.local>
// CHECK-SAME: !xw.ptr<#xw.generic>
// CHECK: xw.local_memory_base {{.*}}xw.global = @scratch
// CHECK: xw.local_memory_base {{.*}}offset = 16 : i64{{.*}}xw.global = @scratch_wide
// CHECK: xw.binary addi {{.*}} overflow<nsw, nuw>
// CHECK: xw.cast intconvert
// CHECK: xw.cast intconvert {{.*}} overflow<nuw> : i32 -> i1
// CHECK: xw.cmpi eq
// CHECK: xw.binary xori
// CHECK: xw.select
// CHECK: scf.if
// CHECK-NOT: llvm.
// CHECK-NOT: {{(^|[^s])cf\.}}

// CHECK-LABEL: func.func @queries_and_math
// CHECK: xw.lane_id : !xw.simd<i32, 16>
// CHECK: xw.subgroup_id : i32
// CHECK: xw.local_id 1 : !xw.simd<i64, 16>
// CHECK: xw.group_id 1 : i64
// CHECK: xw.global_size 1 : i64
// CHECK: xw.local_size 1 : i64
// CHECK: xw.num_groups 1 : i64
// CHECK: xw.launch_grid_size 1 : i64
// CHECK: xw.launch_block_size 1 : i64
// CHECK: xw.ptr_cmp eq
// CHECK: xw.ptr_cmp ne
// CHECK: xw.fadd {{.*}} : !xw.simd<f32, 16>, !xw.simd<f32, 16> -> !xw.simd<f32, 16>
// CHECK: xw.fsub
// CHECK: xw.fmul
// CHECK: xw.atomic_rmw addi {{.*}} : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
// CHECK: xw.atomic_rmw addi {{.*}} : (!xw.simd<i32, 16>, !xw.ptr<#xw.global>) -> (!xw.simd<i32, 16>, !xw.mem.token)
// CHECK-NOT: xw.provisional_cardinality
// CHECK-NOT: arith.
// CHECK-NOT: unrealized_conversion_cast
// CHECK-NOT: llvm.

// WIDTH32: xw.global_id 0 : !xw.simd<i64, 32>
// WIDTH32: xw.load {{.*}} -> (!xw.simd<i32, 32>, !xw.mem.token)
// WIDTH32: xw.lane_id : !xw.simd<i32, 32>
// WIDTH32: xw.local_id 1 : !xw.simd<i64, 32>
// WIDTH32: xw.atomic_rmw addi {{.*}} -> (!xw.simd<i32, 32>, !xw.mem.token)

// CHECK-LABEL: func.func @poison_freeze
// CHECK: %[[POISON:.*]] = ub.poison : i32
// CHECK: xw.freeze %[[POISON]] {{.*}}boundary.note = "kept"
// CHECK: %[[PTR_POISON:.*]] = ub.poison : !xw.ptr<#xw.global>
// CHECK: xw.freeze %[[PTR_POISON]]
// CHECK-NOT: llvm.

// CHECK-LABEL: func.func @mixed
// CHECK: %[[INT_SPLAT:.*]] = xw.splat {{.*}} : i64 -> !xw.simd<i64, 16>
// CHECK: xw.cmpi ult {{.*}}, %[[INT_SPLAT]] : !xw.simd<i64, 16>, !xw.simd<i64, 16> -> !xw.mask<16>
// CHECK: xw.splat {{.*}} : i64 -> !xw.simd<i64, 16>
// CHECK: xw.cmpi ugt {{.*}} : !xw.simd<i64, 16>, !xw.simd<i64, 16> -> !xw.mask<16>
// CHECK: xw.splat {{.*}} : !xw.ptr<#xw.global> -> !xw.simd<!xw.ptr<#xw.global>, 16>
// CHECK: xw.ptr_cmp eq {{.*}} -> !xw.mask<16>
// CHECK: xw.splat {{.*}} : f32 -> !xw.simd<f32, 16>
// CHECK: xw.cmpf olt {{.*}} -> !xw.mask<16>
// CHECK: xw.select {{.*}} : !xw.mask<16>, !xw.simd<i64, 16>
// CHECK: xw.select {{.*}} : !xw.mask<16>, !xw.simd<!xw.ptr<#xw.global>, 16>
// CHECK: xw.select {{.*}} : !xw.mask<16>, !xw.simd<f32, 16>

// CHECK-LABEL: func.func @divergent_if
// CHECK: xw.where {{.*}} {
// CHECK: xw.yield
// CHECK: } otherwise {
// CHECK: xw.yield
// CHECK-NOT: scf.if

// CHECK-LABEL: func.func @mixed_divergent_if
// CHECK: xw.where
// CHECK: xw.yield {{.*}} : !xw.simd<i32, 16>
// CHECK: } otherwise {
// CHECK: %[[FALLBACK:.*]] = xw.splat {{.*}} : i32 -> !xw.simd<i32, 16>
// CHECK: xw.yield %[[FALLBACK]] : !xw.simd<i32, 16>
// CHECK: } : !xw.mask<16> -> !xw.simd<i32, 16>

// CHECK-LABEL: func.func @one_sided_divergent
// CHECK: xw.where {{.*}} {
// CHECK: xw.store
// CHECK: } otherwise {
// CHECK-NOT: scf.if
// CHECK-NOT: xw.imported
// CHECK-NOT: xw.imported_llvm_metadata
// CHECK-NOT: gep_flags
