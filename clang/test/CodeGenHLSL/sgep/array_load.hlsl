// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -fexperimental-emit-sgep -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fexperimental-emit-sgep -o - %s | FileCheck %s

void foo() {
// CHECK: %array = alloca [3 x i32], align 4
  uint array[3] = { 0, 1, 2 };

// CHECK: %[[#PTR:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x i32]) %array, {{i32|i64}} 2)
// CHECK: load i32, ptr %[[#PTR]], align 4
  uint tmp = array[2];
}

struct S {
  uint a;
  uint b;
};

void bar() {
// CHECK: %array = alloca [3 x %struct.S], align 1
  S array[3] = { { 0, 1 }, { 2, 3 }, { 3, 4 } };

// CHECK: %[[#A:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([3 x %struct.S]) %array, {{i32|i64}} 2)
// CHECK: %[[#B:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %[[#A]], {{i32|i64}} 1)
// CHECK: load i32, ptr %[[#B]], align 1
  uint tmp = array[2].b;
}

struct S2 {
  uint a;
  S b;
  uint c;
};

void baz() {
// CHECK: %array = alloca [2 x %struct.S2], align 1
  S2 array[2] = { { 0, { 1, 2 }, 3 }, { 4, { 5, 6 }, 7 } };

// CHECK: %[[#A:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype([2 x %struct.S2]) %array, {{i32|i64}} 1)
// CHECK: %[[#B:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S2) %[[#A]], {{i32|i64}} 1)
// CHECK: %[[#C:]] = call ptr (ptr, ...) @llvm.structured.gep.p0(ptr elementtype(%struct.S) %[[#B]], {{i32|i64}} 0)
// CHECK: load i32, ptr %[[#C]], align 1
  uint tmp = array[1].b.a;
}
