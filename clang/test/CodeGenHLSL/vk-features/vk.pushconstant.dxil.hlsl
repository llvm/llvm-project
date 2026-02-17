// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-compute -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct S {
  uint value;
};
// CHECK: %struct.S = type { i32 }

// When targeting DXIL, the attribute is ignored, meaning this variable
// is part of the implicit cbuffer.
[[vk::push_constant]] S buffer;
// CHECK: @buffer = external hidden addrspace(2) global %struct.S, align 1

[numthreads(1, 1, 1)]
void main() {
  uint32_t v = buffer.value;
// CHECK:  %[[#REG:]] = load i32, ptr addrspace(2) @buffer, align 4
// CHECK:             store i32 %[[#REG]], ptr %v, align 4
}
