// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl -ast-dump -o - %s | FileCheck %s --check-prefix=CHECK-VK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-compute -x hlsl -ast-dump -o - %s | FileCheck %s --check-prefix=CHECK-DX

struct S {
  int value;
};

[[vk::push_constant]] S PC;
// CHECK-VK:      VarDecl 0x[[A:[0-9a-f]+]] <line:8:23, col:25> col:25 PC 'hlsl_push_constant S'
// CHECK-VK-NEXT: HLSLVkPushConstantAttr 0x[[A:[0-9a-f]+]] <col:3, col:7>

// CHECK-DX:      VarDecl 0x[[A:[0-9a-f]+]] <line:8:23, col:25> col:25 PC 'hlsl_constant S'
// CHECK-DX-NEXT: HLSLVkPushConstantAttr 0x[[A:[0-9a-f]+]] <col:3, col:7>

[numthreads(1, 1, 1)]
void main() { }
