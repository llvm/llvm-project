// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-compute -std=hlsl202x -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s --check-prefixes=CHECK

struct S {
  static int Value;
};

int S::Value = 1;
// CHECK: @_ZN1S5ValueE = global i32 1, align 4

[shader("compute")]
[numthreads(1,1,1)]
void main() {
  S s;
  int value1, value2;
// CHECK:      %s = alloca %struct.S, align 1
// CHECK: %value1 = alloca i32, align 4
// CHECK: %value2 = alloca i32, align 4

// CHECK: [[tmp:%.*]] = load i32, ptr @_ZN1S5ValueE, align 4
// CHECK: store i32 [[tmp]], ptr %value1, align 4
  value1 = S::Value;

// CHECK: [[tmp:%.*]] = load i32, ptr @_ZN1S5ValueE, align 4
// CHECK: store i32 [[tmp]], ptr %value2, align 4
  value2 = s.Value;
}
