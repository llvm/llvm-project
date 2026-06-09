// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -fexperimental-logical-pointer -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fexperimental-logical-pointer -o - %s | FileCheck %s

[shader("compute")]
[numthreads(1,1,1)]
void foo() {
// CHECK: %array = call elementtype([10 x i32]) ptr @llvm.structured.alloca.p0()
  uint array[10];

// CHECK: %[[FOO_PTR:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([10 x i32]) %array, <1 x i32> splat (i32 3), {{i32|i64}} 2)
// CHECK: store i32 10, ptr %[[FOO_PTR]], align 4
  array[2] = 10;
}

struct S {
  uint a;
  uint b;
};

void bar() {
// CHECK: %array = call elementtype([3 x %struct.S]) ptr @llvm.structured.alloca.p0()
  S array[3] = { { 0, 1 }, { 2, 3 }, { 3, 4 } };

// CHECK: %[[BAR_A:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([3 x %struct.S]) %array, <1 x i32> splat (i32 3), {{i32|i64}} 2)
// CHECK: %[[BAR_B:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%struct.S) %[[BAR_A]], <1 x i32> splat (i32 3), {{i32|i64}} 1)
// CHECK: store i32 10, ptr %[[BAR_B]], align 1

  array[2].b = 10;
}

struct S2 {
  uint a;
  S b;
  uint c;
};

void baz() {
// CHECK: %array = call elementtype([2 x %struct.S2]) ptr @llvm.structured.alloca.p0()
  S2 array[2] = { { 0, { 1, 2 }, 3 }, { 4, { 5, 6 }, 7 } };

// CHECK: %[[BAZ_A:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([2 x %struct.S2]) %array, <1 x i32> splat (i32 3), {{i32|i64}} 1)
// CHECK: %[[BAZ_B:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%struct.S2) %[[BAZ_A]], <1 x i32> splat (i32 3), {{i32|i64}} 1)
// CHECK: %[[BAZ_C:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype(%struct.S) %[[BAZ_B]], <1 x i32> splat (i32 3), {{i32|i64}} 0)
// CHECK: store i32 10, ptr %[[BAZ_C]], align 1

  array[1].b.a = 10;
}
