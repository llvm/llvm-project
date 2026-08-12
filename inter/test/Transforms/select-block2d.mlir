// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @block2d(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %width = xw.constant 128 : i32
    %height = xw.constant 128 : i32
    %pitch = xw.constant 128 : i32
    %x = xw.constant 0 : i32
    %y = xw.constant 16 : i32
    %root = xw.token : !xw.mem.token
    %prefetched = xw.block2d_prefetch %base[%x, %y] surface (%width, %height, %pitch) after %root {block_height = 8 : i64, block_width = 16 : i64, blocks = 1 : i64, element_bits = 16 : i64} : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32, !xw.mem.token) -> !xw.mem.token
    %value, %loaded = xw.block2d_read %base[%x, %y] surface (%width, %height, %pitch) after %prefetched {block_height = 8 : i64, block_width = 16 : i64, blocks = 1 : i64, element_bits = 16 : i64} : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32, !xw.mem.token) -> (!xw.simd<vector<8xi16>, 16>, !xw.mem.token)
    %write_value = xw.constant dense<0> : vector<8xi32> -> !xw.simd<vector<8xi32>, 16>
    %stored = xw.block2d_write %write_value -> %base[%x, %y] surface (%width, %height, %pitch) after %loaded {block_height = 8 : i64, block_width = 16 : i64, blocks = 1 : i64, element_bits = 32 : i64} : (!xw.simd<vector<8xi32>, 16>, !xw.ptr<#xw.global>, i32, i32, i32, i32, i32, !xw.mem.token) -> !xw.mem.token
    return
  }
}

// CHECK-LABEL: func.func @block2d
// CHECK: xemachine.imm 1823 : i32
// CHECK: xemachine.update_tuple
// CHECK: xemachine.send ugm {{.*}}desc = 34079235
// CHECK: xemachine.imm 1823 : i32
// CHECK: xemachine.update_tuple
// CHECK: xemachine.send ugm {{.*}}desc = 37749251
// CHECK: xemachine.imm 1855 : i32
// CHECK: xemachine.update_tuple
// CHECK: xemachine.send ugm {{.*}}data {{.*}}desc = 33555463
// CHECK: xemachine.eot
