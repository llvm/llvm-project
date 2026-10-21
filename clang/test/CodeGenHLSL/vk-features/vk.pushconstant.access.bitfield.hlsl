// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  uint32_t a : 1;
  uint32_t b : 1;
};
// CHECK: %struct.S = type { i8 }

[[vk::push_constant]] S buffer;
// CHECK: @buffer = external hidden addrspace(13) externally_initialized global %struct.S, align 1

[numthreads(1, 1, 1)]
void main() {
  uint32_t v = buffer.b;
// CHECK:  %bf.load = load i8, ptr addrspace(13) @buffer, align 1
// CHECK:  %bf.lshr = lshr i8 %bf.load, 1
// CHECK: %bf.clear = and i8 %bf.lshr, 1
// CHECK:  %bf.cast = zext i8 %bf.clear to i32
// CHECK:             store i32 %bf.cast
}
