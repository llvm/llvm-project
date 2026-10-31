// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s
// RUN: inter-opt %s --inter-select-to-machine -o %t
// RUN: inter-opt %t --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' | FileCheck %s --check-prefix=ALLOC

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
// CHECK: xemachine.imm 1807 : i32
// CHECK: xemachine.update_tuple
// CHECK: %[[PREFETCH:.*]], %[[PREFETCH_TOKEN:.*]] = xemachine.send ugm {{.*}}desc = 34079235
// CHECK: %[[ISSUED:.*]] = xemachine.after %[[PREFETCH_TOKEN]]
// CHECK: xemachine.imm 1807 : i32
// CHECK: xemachine.update_tuple
// CHECK: xemachine.send ugm {{.*}}dep %[[ISSUED]] {{.*}}desc = 37749251
// CHECK: xemachine.imm 1807 : i32
// CHECK: xemachine.update_tuple
// CHECK: xemachine.send ugm {{.*}}data {{.*}}desc = 33555463
// CHECK: xemachine.eot

// ALLOC-LABEL: func.func @block2d
// ALLOC: [[SHAPE:%.*]] = xemachine.mov {{.*}}dstSub = 7{{.*}}-> !xemachine.reg<1, [[PAYLOAD:[0-9]+]]>
// ALLOC: [[X:%.*]] = xemachine.mov {{.*}}dstSub = 5{{.*}}-> !xemachine.reg<1, [[PAYLOAD]]>
// ALLOC: [[Y:%.*]] = xemachine.mov {{.*}}dstSub = 6{{.*}}-> !xemachine.reg<1, [[PAYLOAD]]>
// ALLOC: [[WIDTH:%.*]] = xemachine.mov {{.*}}dstSub = 2{{.*}}-> !xemachine.reg<1, [[PAYLOAD]]>
// ALLOC: [[HEIGHT:%.*]] = xemachine.mov {{.*}}dstSub = 3{{.*}}-> !xemachine.reg<1, [[PAYLOAD]]>
// ALLOC: [[PITCH:%.*]] = xemachine.mov {{.*}}dstSub = 4{{.*}}-> !xemachine.reg<1, [[PAYLOAD]]>
// ALLOC: [[ADDRESS:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-value"{{.*}}-> !xemachine.reg<2, [[PAYLOAD]]>
// ALLOC-NEXT: [[TUPLE:%.*]] = xemachine.update_tuple {{%.*}}, [[ADDRESS]], [[WIDTH]], [[HEIGHT]], [[PITCH]], [[X]], [[Y]], [[SHAPE]]
// ALLOC-NEXT: xemachine.send ugm [[TUPLE]]
