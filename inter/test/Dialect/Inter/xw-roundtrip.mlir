// RUN: inter-opt %s | inter-opt | FileCheck %s

// CHECK-LABEL: func.func @surface
// CHECK-DAG: #xw.private
// CHECK-DAG: #xw.global
// CHECK-DAG: #xw.constant
// CHECK-DAG: #xw.local
// CHECK-DAG: #xw.generic
// CHECK-DAG: !xw.mem.token
// CHECK-DAG: xw.freeze {{.*}} : !xw.simd<i32, 16>
func.func @surface(%u: i32, %private: !xw.ptr<#xw.private>,
                   %p: !xw.ptr<#xw.global>,
                   %constant: !xw.ptr<#xw.constant>,
                   %lp: !xw.ptr<#xw.local>,
                   %v: !xw.simd<i32, 8>, %m: !xw.mask<8>)
    attributes {xw.simd_width = 32 : i64} {
  %c = xw.constant 7 : i32 -> !xw.simd<i32, 8>
  %s = xw.splat %u : i32 -> !xw.simd<i32, 8>
  %first = xw.read_first %s : !xw.simd<i32, 8> -> i32
  %expanded = xw.expand %v : !xw.simd<i32, 8> -> !xw.simd<i32, 16>
  %frozen = xw.freeze %expanded : !xw.simd<i32, 16>
  %freeze_use = xw.binary addi %frozen, %frozen overflow<nsw, nuw>
      : !xw.simd<i32, 16>, !xw.simd<i32, 16> -> !xw.simd<i32, 16>
  %sum = xw.binary addi %s, %u
      : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
  %wide = xw.cast intconvert %sum policy {extension = #xw.cast_extension<zero>}
      : !xw.simd<i32, 8> -> !xw.simd<i64, 8>
  %narrow = xw.cast intconvert %wide overflow<nuw>
      : !xw.simd<i64, 8> -> !xw.simd<i32, 8>
  %cmp = xw.cmpi slt %sum, %c
      : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.mask<8>
  %selected = xw.select %cmp, %sum, %c
      : !xw.mask<8>, !xw.simd<i32, 8>
  %and = xw.mask_and %cmp, %m : !xw.mask<8>
  %not = xw.mask_not %and : !xw.mask<8>
  %bits = xw.ballot %not : !xw.mask<8> -> i8
  %packed = xw.pack %sum, %selected
      : !xw.simd<i32, 8>, !xw.simd<i32, 8>
      -> !xw.simd<vector<2xi32>, 8>
  %reinterpreted = xw.bitcast %packed : !xw.simd<vector<2xi32>, 8>
      -> !xw.simd<vector<2xf32>, 8>
  %element = xw.extract %packed[0] : !xw.simd<vector<2xi32>, 8>
      -> !xw.simd<i32, 8>
  %lane = xw.lane_id : !xw.simd<i32, 8>
  %subgroup = xw.subgroup_id : i32
  %gid = xw.global_id 0 : !xw.simd<i32, 8>
  %group = xw.group_id 0 : i32
  %grid = xw.launch_grid_size 0 : i32
  %shuffled = xw.shuffle %sum from %u
      : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
  %pi = xw.ptr_to_int %p : !xw.ptr<#xw.global> -> i64
  %p2 = xw.int_to_ptr %pi : i64 -> !xw.ptr<#xw.global>
  %generic = xw.addrspace_cast %p2
      : !xw.ptr<#xw.global> -> !xw.ptr<#xw.generic>
  %null = xw.null : !xw.ptr<#xw.global>
  %isnull = xw.ptr_cmp eq %p, %null
      : !xw.ptr<#xw.global>, !xw.ptr<#xw.global> -> i1
  %root = xw.token : !xw.mem.token
  %value, %loaded = xw.load %p after %root
      : (!xw.ptr<#xw.global>, !xw.mem.token)
      -> (!xw.simd<i32, 8>, !xw.mem.token)
  %stored = xw.store %value -> %p after %loaded
      : (!xw.simd<i32, 8>, !xw.ptr<#xw.global>, !xw.mem.token)
      -> !xw.mem.token
  %old, %atomic = xw.atomic_rmw addi %value, %p after %stored
      : (!xw.simd<i32, 8>, !xw.ptr<#xw.global>, !xw.mem.token)
      -> (!xw.simd<i32, 8>, !xw.mem.token)
  %issued = xw.issue_token %atomic
      : !xw.mem.token -> !xw.mem.token
  %after = xw.after %issued : !xw.mem.token -> !xw.mem.token
  %joined = xw.join %after, %loaded
      : !xw.mem.token, !xw.mem.token -> !xw.mem.token
  %barrier = xw.barrier %joined
      : !xw.mem.token -> !xw.mem.token
  %local = xw.local_memory_base : !xw.ptr<#xw.local>
  %allocation = xw.alloc() {bytesize = 64 : i64, align = 16 : i64}
      : !xw.ptr<#xw.local>
  %released = xw.alloc_release %allocation after %barrier
      : (!xw.ptr<#xw.local>, !xw.mem.token) -> !xw.mem.token
  xw.where %m {
    xw.yield
  } : !xw.mask<8>
  return
}

// CHECK-NOT: !llvm.ptr
// CHECK-NOT: !xemachine.mem.token
