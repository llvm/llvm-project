// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @atomic_before_id(%out: !xw.ptr<#xw.global>,
                              %counter: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 24>,
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 32>
      ],
      xw.simd_width = 32 : i32} {
    %one = xw.constant 1 : i32
    %ones = xw.splat %one : i32 -> !xw.simd<i32, 32>
    %root = xw.token : !xw.mem.token
    %old, %atomic = xw.atomic_rmw addi %ones, %counter after %root : (!xw.simd<i32, 32>, !xw.ptr<#xw.global>, !xw.mem.token) -> (!xw.simd<i32, 32>, !xw.mem.token)
    %gid = xw.global_id 0 : !xw.simd<i64, 32>
    %two = xw.constant 2 : i64 -> !xw.simd<i64, 32>
    %offset = xw.binary shli %gid, %two : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.simd<i64, 32>
    %address = xw.ptradd %out, %offset : !xw.ptr<#xw.global>, !xw.simd<i64, 32> -> !xw.simd<!xw.ptr<#xw.global>, 32>
    %stored = xw.store %old -> %address after %atomic : (!xw.simd<i32, 32>, !xw.simd<!xw.ptr<#xw.global>, 32>, !xw.mem.token) -> !xw.mem.token
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @atomic_before_id
// CHECK: xemachine.load_block_a32
// CHECK: xemachine.sync allwr
// CHECK: xemachine.atomic_iadd_a64
// CHECK-COUNT-2: xemachine.mov {{.*}}dstRegion = #xemachine.dstregion<4>
// CHECK-COUNT-4: xemachine.add
