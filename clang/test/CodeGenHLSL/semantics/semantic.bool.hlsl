// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

[shader("vertex")]
bool main(bool b : B) : A {
  return b;
}

// DXIL signatures represent bool as i32, while the entry implementation uses
// the i1 value representation.
// CHECK-LABEL: define void @main()
// CHECK: %[[INPUT:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: %[[BOOL:.*]] = icmp ne i32 %[[INPUT]], 0
// CHECK: %[[RESULT:.*]] = call i1 @_Z4mainb(i1 %[[BOOL]])
// CHECK: %[[OUTPUT:.*]] = zext i1 %[[RESULT]] to i32
// CHECK: call void @llvm.dx.store.output.i32(i32 0, i32 0, i8 0, i32 %[[OUTPUT]])

[shader("vertex")]
bool other(bool b : D) : C {
  return b;
}

// Signature element IDs are local to each entry point.
// CHECK-LABEL: define void @other()
// CHECK: call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK: call void @llvm.dx.store.output.i32(i32 0, i32 0, i8 0, i32 %{{.*}})
