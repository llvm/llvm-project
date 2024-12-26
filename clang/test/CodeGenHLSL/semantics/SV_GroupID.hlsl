// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

// Make sure SV_GroupID translated into dx.group.id.

// CHECK:  define void @foo()
// CHECK:  %[[#ID:]] = call i32 @llvm.dx.group.id(i32 0)
// CHECK:  call void @{{.*}}foo{{.*}}(i32 %[[#ID]])
[shader("compute")]
[numthreads(8,8,1)]
void foo(uint Idx : SV_GroupID) {}

// CHECK:  define void @bar()
// CHECK:  %[[#ID_X:]] = call i32 @llvm.dx.group.id(i32 0)
// CHECK:  %[[#ID_X_:]] = insertelement <2 x i32> poison, i32 %[[#ID_X]], i64 0
// CHECK:  %[[#ID_Y:]] = call i32 @llvm.dx.group.id(i32 1)
// CHECK:  %[[#ID_XY:]] = insertelement <2 x i32> %[[#ID_X_]], i32 %[[#ID_Y]], i64 1
// CHECK:  call void @{{.*}}bar{{.*}}(<2 x i32> %[[#ID_XY]])
[shader("compute")]
[numthreads(8,8,1)]
void bar(uint2 Idx : SV_GroupID) {}

// CHECK:  define void @test()
// CHECK:  %[[#ID_X:]] = call i32 @llvm.dx.group.id(i32 0)
// CHECK:  %[[#ID_X_:]] = insertelement <3 x i32> poison, i32 %[[#ID_X]], i64 0
// CHECK:  %[[#ID_Y:]] = call i32 @llvm.dx.group.id(i32 1)
// CHECK:  %[[#ID_XY:]] = insertelement <3 x i32> %[[#ID_X_]], i32 %[[#ID_Y]], i64 1
// CHECK:  %[[#ID_Z:]] = call i32 @llvm.dx.group.id(i32 2)
// CHECK:  %[[#ID_XYZ:]] = insertelement <3 x i32> %[[#ID_XY]], i32 %[[#ID_Z]], i64 2
// CHECK:  call void @{{.*}}test{{.*}}(<3 x i32> %[[#ID_XYZ]])
[shader("compute")]
[numthreads(8,8,1)]
void test(uint3 Idx : SV_GroupID) {}
