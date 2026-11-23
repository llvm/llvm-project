// RUN: %clang_cc1 -triple spirv-pc-vulkan-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S
{
  const static uint a = 1;
  uint b;
};
// CHECK: %struct.S = type { i32 }

[[vk::push_constant]] S s;
// CHECK: @s = external hidden addrspace(13) externally_initialized global %struct.S, align 1

[numthreads(1,1,1)]
void main()
{
  uint32_t v = s.b;
  // CHECK: %[[#TMP:]] = load i32, ptr addrspace(13) @s, align 1
  // CHECK:              store i32 %[[#TMP]], ptr %v, align 4

  uint32_t w = S::a;
  // CHECK:   store i32 1, ptr %w, align 4

  uint32_t x = s.a;
  // CHECK:   store i32 1, ptr %x, align 4
}
