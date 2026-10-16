// RUN: inter-opt %s --inter-refine-distribution='simd-width=16' | FileCheck %s
// RUN: inter-opt %s --inter-refine-distribution='simd-width=16' --inter-refine-distribution='simd-width=16' --verify-each -o /dev/null

func.func @width8() attributes {xw.simd_width = 8 : i64} {
  %lane = xw.lane_id : !xw.simd<i32, 8>
  return
}

func.func @width16() attributes {xw.simd_width = 16 : i64} {
  %gid = xw.global_id 0 {xw.provisional_cardinality = 16 : i32}
      : !xw.simd<i32, 8>
  return
}

func.func @width32() attributes {xw.simd_width = 32 : i64} {
  %lid = xw.local_id 0 {xw.provisional_cardinality = 16 : i32}
      : !xw.simd<i32, 16>
  return
}

// CHECK-LABEL: func.func @width8
// CHECK: xw.lane_id : !xw.simd<i32, 8>
// CHECK-LABEL: func.func @width16
// CHECK: xw.global_id 0 : !xw.simd<i32, 16>
// CHECK-LABEL: func.func @width32
// CHECK: xw.local_id 0 : !xw.simd<i32, 32>
// CHECK-NOT: xw.provisional_cardinality
// CHECK-NOT: xw.distribution

func.func @uniform_and_adapters(%arg: i32)
    attributes {xw.simd_width = 32 : i64} {
  %constant = xw.constant 1 : i32 -> !xw.simd<i32, 8>
  %group = xw.group_id 0 : i32
  %global_size = xw.global_size 0 : i32
  %local_size = xw.local_size 0 : i32
  %groups = xw.num_groups 0 : i32
  %grid = xw.launch_grid_size 0 : i32
  %block = xw.launch_block_size 0 : i32
  %null = xw.null : !xw.ptr<#xw.global>
  %base = xw.local_memory_base : !xw.ptr<#xw.local>
  %alloc = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %splat = xw.splat %arg : i32 -> !xw.simd<i32, 8>
  %expand = xw.expand %splat : !xw.simd<i32, 8> -> !xw.simd<i32, 16>
  %first = xw.read_first %expand : !xw.simd<i32, 16> -> i32
  %mask = xw.cmpi ne %splat, %splat
      : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.mask<8>
  %bits = xw.ballot %mask : !xw.mask<8> -> i8
  return
}

// CHECK-LABEL: func.func @uniform_and_adapters(
// CHECK-SAME: %[[ARG:.*]]: i32
// CHECK: xw.constant 1 : i32
// CHECK: xw.group_id 0 : i32
// CHECK: xw.global_size 0 : i32
// CHECK: xw.local_size 0 : i32
// CHECK: xw.num_groups 0 : i32
// CHECK: xw.launch_grid_size 0 : i32
// CHECK: xw.launch_block_size 0 : i32
// CHECK: xw.null : !xw.ptr<#xw.global>
// CHECK: xw.local_memory_base : !xw.ptr<#xw.local>
// CHECK: xw.alloc() {{.*}} : !xw.ptr<#xw.local>
// CHECK: %[[SPLAT:.*]] = xw.splat %[[ARG]] : i32 -> !xw.simd<i32, 8>
// CHECK: %[[EXPAND:.*]] = xw.expand %[[SPLAT]] : !xw.simd<i32, 8> -> !xw.simd<i32, 16>
// CHECK: xw.read_first %[[EXPAND]] : !xw.simd<i32, 16> -> i32
// CHECK: xw.ballot {{.*}} : !xw.mask<8> -> i8

func.func @regions(%arg: i32)
    attributes {xw.simd_width = 32 : i64} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %uniform = xw.constant 7 : i32 -> !xw.simd<i32, 8>
  %lane = xw.lane_id : !xw.simd<i32, 32>
  %truth = arith.constant true

  %if_uniform, %if_lane = scf.if %truth
      -> (!xw.simd<i32, 8>, !xw.simd<i32, 32>) {
    scf.yield %uniform, %lane : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  } else {
    scf.yield %uniform, %lane : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  }

  %for_uniform, %for_lane = scf.for %iv = %zero to %one step %one
      iter_args(%u = %if_uniform, %v = %if_lane)
      -> (!xw.simd<i32, 8>, !xw.simd<i32, 32>) {
    scf.yield %u, %v : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  }

  %while_uniform, %while_lane = scf.while
      (%u = %for_uniform, %v = %for_lane)
      : (!xw.simd<i32, 8>, !xw.simd<i32, 32>)
      -> (!xw.simd<i32, 8>, !xw.simd<i32, 32>) {
    scf.condition(%truth) %u, %v
        : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  } do {
  ^bb0(%u: !xw.simd<i32, 8>, %v: !xw.simd<i32, 32>):
    scf.yield %u, %v : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  }

  %mask = xw.cmpi eq %lane, %lane
      : !xw.simd<i32, 32>, !xw.simd<i32, 32> -> !xw.mask<32>
  %where_uniform, %where_lane = xw.where %mask {
    xw.yield %while_uniform, %while_lane
        : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  } otherwise {
    xw.yield %uniform, %lane : !xw.simd<i32, 8>, !xw.simd<i32, 32>
  } : !xw.mask<32> -> !xw.simd<i32, 8>, !xw.simd<i32, 32>
  return
}

// CHECK-LABEL: func.func @regions
// CHECK: {{.*}} = scf.if {{.*}} -> (i32, !xw.simd<i32, 32>)
// CHECK: scf.yield {{.*}} : i32, !xw.simd<i32, 32>
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (i32, !xw.simd<i32, 32>)
// CHECK: scf.yield {{.*}} : i32, !xw.simd<i32, 32>
// CHECK: scf.while {{.*}} : (i32, !xw.simd<i32, 32>) -> (i32, !xw.simd<i32, 32>)
// CHECK: ^bb0(%{{.*}}: i32, %{{.*}}: !xw.simd<i32, 32>)
// CHECK: {{.*}} = xw.where
// CHECK: xw.yield {{.*}} : i32, !xw.simd<i32, 32>
// CHECK: } : !xw.mask<32> -> i32, !xw.simd<i32, 32>

func.func @memory(%ptr: !xw.ptr<#xw.global>, %value: i32)
    attributes {xw.simd_width = 32 : i64} {
  %root = xw.token : !xw.mem.token
  %loaded, %read = xw.load %ptr after %root
      : (!xw.ptr<#xw.global>, !xw.mem.token)
      -> (!xw.simd<i32, 16>, !xw.mem.token)
  %old, %atomic = xw.atomic_rmw addi %value, %ptr after %read
      : (i32, !xw.ptr<#xw.global>, !xw.mem.token)
      -> (i32, !xw.mem.token)
  return
}

// CHECK-LABEL: func.func @memory(
// CHECK-SAME: %[[PTR:.*]]: !xw.ptr<#xw.global>, %[[VALUE:.*]]: i32
// CHECK: %[[ROOT:.*]] = xw.token : !xw.mem.token
// CHECK: %{{.*}}, %[[READ:.*]] = xw.load %[[PTR]] after %[[ROOT]] : (!xw.ptr<#xw.global>, !xw.mem.token) -> (!xw.simd<i32, 32>, !xw.mem.token)
// CHECK: %[[SPLAT:.*]] = xw.splat %[[VALUE]] : i32 -> !xw.simd<i32, 32>
// CHECK: %{{.*}}, %{{.*}} = xw.atomic_rmw addi %[[SPLAT]], %[[PTR]] after %[[READ]] : (!xw.simd<i32, 32>, !xw.ptr<#xw.global>, !xw.mem.token) -> (!xw.simd<i32, 32>, !xw.mem.token)

func.func @kernel_abi(%arg: i32, %ptr: !xw.ptr<#xw.global>)
    attributes {xemachine.kernel, xw.simd_width = 16 : i64} {
  %splat = xw.splat %arg : i32 -> !xw.simd<i32, 16>
  return
}

// CHECK-LABEL: func.func @kernel_abi(%{{.*}}: i32, %{{.*}}: !xw.ptr<#xw.global>)
// CHECK: xw.splat {{.*}} : i32 -> !xw.simd<i32, 16>

func.func @resultless_controls(%condition: i1, %mask: !xw.mask<16>)
    attributes {xw.simd_width = 16 : i64} {
  scf.if %condition {
  }
  xw.where %mask {
    xw.yield
  } : !xw.mask<16>
  return
}

// CHECK-LABEL: func.func @resultless_controls
// CHECK: scf.if
// CHECK: xw.where

func.func @simd_only_consumers(%lane: i32)
    attributes {xw.simd_width = 16 : i64} {
  %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 8>
  %sum = xw.fadd %one, %one
      : !xw.simd<f32, 8>, !xw.simd<f32, 8> -> !xw.simd<f32, 8>
  %expanded = xw.expand %sum : !xw.simd<f32, 8> -> !xw.simd<f32, 16>
  %first = xw.read_first %expanded : !xw.simd<f32, 16> -> f32
  %shuffled = xw.shuffle %expanded from %lane
      : !xw.simd<f32, 16>, i32 -> !xw.simd<f32, 16>
  return
}

// CHECK-LABEL: func.func @simd_only_consumers
// CHECK: xw.fadd {{.*}} : !xw.simd<f32, 8>, !xw.simd<f32, 8> -> !xw.simd<f32, 8>
// CHECK: xw.expand {{.*}} : !xw.simd<f32, 8> -> !xw.simd<f32, 16>
// CHECK: xw.read_first {{.*}} : !xw.simd<f32, 16> -> f32
// CHECK: xw.shuffle {{.*}} : !xw.simd<f32, 16>, i32 -> !xw.simd<f32, 16>

func.func private @boundary_callee(!xw.simd<i32, 8>) -> !xw.simd<i32, 8>

func.func @boundary_caller(%arg: !xw.simd<i32, 8>) -> !xw.simd<i32, 8>
    attributes {xw.simd_width = 16 : i64} {
  %result = func.call @boundary_callee(%arg)
      : (!xw.simd<i32, 8>) -> !xw.simd<i32, 8>
  return %result : !xw.simd<i32, 8>
}

// CHECK-LABEL: func.func @boundary_caller(%{{.*}}: !xw.simd<i32, 8>) -> !xw.simd<i32, 8>
// CHECK: call @boundary_callee({{.*}}) : (!xw.simd<i32, 8>) -> !xw.simd<i32, 8>
// CHECK: return {{.*}} : !xw.simd<i32, 8>

func.func @poison_and_freeze()
    attributes {xw.simd_width = 16 : i64} {
  %poison8 = ub.poison : !xw.simd<i32, 8>
  %poison_bare = ub.poison : i32
  %frozen8 = xw.freeze %poison8 : !xw.simd<i32, 8>
  %frozen_bare = xw.freeze %poison_bare : i32
  return
}

// CHECK-LABEL: func.func @poison_and_freeze
// CHECK: %[[P8:.*]] = ub.poison : !xw.simd<i32, 8>
// CHECK: %[[PB:.*]] = ub.poison : i32
// CHECK: xw.freeze %[[P8]] : !xw.simd<i32, 8>
// CHECK: xw.freeze %[[PB]] : i32
