// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -fexperimental-logical-pointer -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fexperimental-logical-pointer -o - %s | FileCheck %s

void foo() {
// CHECK: %array = call elementtype([3 x i32]) ptr @llvm.structured.alloca.p0()
  uint array[3] = { 0, 1, 2 };

// CHECK: %[[FOO_PTR:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([3 x i32]) %array, <1 x i32> splat (i32 3), {{i32|i64}} 2)
// CHECK: load i32, ptr %[[FOO_PTR]], align 4
  uint tmp = array[2];
}

void signed_idx(int i) {
// CHECK: %array = call elementtype([3 x i32]) ptr @llvm.structured.alloca.p0()
  uint array[3] = { 0, 1, 2 };

// CHECK: %[[SIGNED_PTR:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([3 x i32]) %array, <1 x i32> splat (i32 1), {{i32|i64}} %{{.*}})
// CHECK: load i32, ptr %[[SIGNED_PTR]], align 4
  uint tmp = array[i];
}

void unsigned_idx(uint i) {
// CHECK: %array = call elementtype([3 x i32]) ptr @llvm.structured.alloca.p0()
  uint array[3] = { 0, 1, 2 };

// CHECK: %[[UNSIGNED_PTR:.*]] = call ptr (ptr, <1 x i32>, ...) @llvm.structured.gep.p0.v1i32(ptr elementtype([3 x i32]) %array, <1 x i32> splat (i32 5), {{i32|i64}} %{{.*}})
// CHECK: load i32, ptr %[[UNSIGNED_PTR]], align 4
  uint tmp = array[i];
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
// CHECK: load i32, ptr %[[BAR_B]], align 1
  uint tmp = array[2].b;
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
// CHECK: load i32, ptr %[[BAZ_C]], align 1
  uint tmp = array[1].b.a;
}
