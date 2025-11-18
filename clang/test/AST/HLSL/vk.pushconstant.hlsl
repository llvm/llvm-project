// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl -ast-dump -o - %s | FileCheck %s

struct S {
  int value;
};

[[vk::push_constant]] S PC;
// CHECK:      VarDecl 0x[[A:[0-9a-f]+]] <line:7:23, col:25> col:25 PC 'hlsl_push_constant S'
// CHECK-NEXT: HLSLVkPushConstantAttr 0x[[A:[0-9a-f]+]] <col:3, col:7>

[numthreads(1, 1, 1)]
void main() { }
