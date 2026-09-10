// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @vadd(%a: !xw.ptr<#xw.global>, %b: !xw.ptr<#xw.global>,
                  %out: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 24>,
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 32>,
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 40>
      ],
      xw.simd_width = 32 : i32} {
    %gid = xw.global_id 0 : !xw.simd<i64, 32>
    %two = xw.constant 2 : i64 -> !xw.simd<i64, 32>
    %offset = xw.binary shli %gid, %two : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.simd<i64, 32>
    %pa = xw.ptradd %a, %offset : !xw.ptr<#xw.global>, !xw.simd<i64, 32> -> !xw.simd<!xw.ptr<#xw.global>, 32>
    %pb = xw.ptradd %b, %offset : !xw.ptr<#xw.global>, !xw.simd<i64, 32> -> !xw.simd<!xw.ptr<#xw.global>, 32>
    %po = xw.ptradd %out, %offset : !xw.ptr<#xw.global>, !xw.simd<i64, 32> -> !xw.simd<!xw.ptr<#xw.global>, 32>
    %root = xw.token : !xw.mem.token
    %va, %ta = xw.load %pa after %root : (!xw.simd<!xw.ptr<#xw.global>, 32>, !xw.mem.token) -> (!xw.simd<i32, 32>, !xw.mem.token)
    %vb, %tb = xw.load %pb after %root : (!xw.simd<!xw.ptr<#xw.global>, 32>, !xw.mem.token) -> (!xw.simd<i32, 32>, !xw.mem.token)
    %sum = xw.binary addi %va, %vb : !xw.simd<i32, 32>, !xw.simd<i32, 32> -> !xw.simd<i32, 32>
    %loads = xw.join %ta, %tb : !xw.mem.token, !xw.mem.token -> !xw.mem.token
    %stored = xw.store %sum -> %po after %loads : (!xw.simd<i32, 32>, !xw.simd<!xw.ptr<#xw.global>, 32>, !xw.mem.token) -> !xw.mem.token
    return
  }

  func.func @id_payload_sizes() attributes {
      xemachine.kernel,
      xemachine.kernel_args = [],
      xw.simd_width = 32 : i32} {
    %y = xw.local_id 1 : !xw.simd<i64, 32>
    %z = xw.local_id 2 : !xw.simd<i64, 32>
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @vadd
// CHECK-SAME: xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 24>
// CHECK-SAME: xemachine.simd_size = 32 : i32
// CHECK: xemachine.payload_prologue {
// CHECK: xemachine.load_block_a32
// CHECK: xemachine.payload_prologue_end
// CHECK: }
// CHECK-COUNT-2: xemachine.mov {{.*}}dstRegion = #xemachine.dstregion<4>
// CHECK-COUNT-4: xemachine.add
// CHECK-COUNT-2: xemachine.load_a64
// CHECK: xemachine.add
// CHECK: xemachine.store_a64
// CHECK: xemachine.eot

// CHECK-LABEL: func.func @id_payload_sizes
// CHECK-SAME: xemachine.per_thread_payload_size = 192 : i32
// CHECK: xemachine.load_block_a32 {{.*}}words = 32
// CHECK: xemachine.load_block_a32 {{.*}}words = 16
