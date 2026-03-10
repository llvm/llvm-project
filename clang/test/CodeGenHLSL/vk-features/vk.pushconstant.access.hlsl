// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  uint a;
};
// CHECK: %struct.S = type { i32 }

[[vk::push_constant]] S buffer;
// CHECK: @buffer = external hidden addrspace(13) externally_initialized global %struct.S, align 1

[numthreads(1, 1, 1)]
void main() {
  uint32_t v = buffer.a;
// CHECK:  %[[#REG:]] = load i32, ptr addrspace(13) @buffer, align 1
// CHECK:               store i32 %[[#REG]], ptr %v, align 4
}
