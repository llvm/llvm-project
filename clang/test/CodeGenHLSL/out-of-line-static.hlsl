// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-compute -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK,SPIRV

struct S {
  static int Value;
};

int S::Value = 1;
// DXIL: @_ZN1S5ValueE = hidden global i32 1, align 4
// SPIRV: @_ZN1S5ValueE = hidden addrspace(10) global i32 1, align 4

[shader("compute")]
[numthreads(1,1,1)]
void main() {
  S s;
  int value1, value2;
// CHECK:      %s = alloca %struct.S, align 1
// CHECK: %value1 = alloca i32, align 4
// CHECK: %value2 = alloca i32, align 4

// DXIL: [[tmp:%.*]] = load i32, ptr @_ZN1S5ValueE, align 4
// SPIRV: [[tmp:%.*]] = load i32, ptr addrspace(10) @_ZN1S5ValueE, align 4
// CHECK: store i32 [[tmp]], ptr %value1, align 4
  value1 = S::Value;

// DXIL: [[tmp:%.*]] = load i32, ptr @_ZN1S5ValueE, align 4
// SPIRV: [[tmp:%.*]] = load i32, ptr addrspace(10) @_ZN1S5ValueE, align 4
// CHECK: store i32 [[tmp]], ptr %value2, align 4
  value2 = s.Value;
}
