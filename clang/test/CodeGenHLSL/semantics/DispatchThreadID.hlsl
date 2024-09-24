// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv

// Make sure SV_DispatchThreadID translated into dx.thread.id.

// CHECK:       define void @foo()
// CHECK-DXIL:  %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK-SPIRV: %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK:       call void @{{.*}}foo{{.*}}(i32 %[[#ID]])
[shader("compute")]
[numthreads(8,8,1)]
void foo(uint Idx : SV_DispatchThreadID) {}

// CHECK:       define void @bar()
// CHECK:       %[[#ID_X:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK:       %[[#ID_X_:]] = insertelement <2 x i32> poison, i32 %[[#ID_X]], i64 0
// CHECK:       %[[#ID_Y:]] = call i32 @llvm.[[TARGET]].thread.id(i32 1)
// CHECK:       %[[#ID_XY:]] = insertelement <2 x i32> %[[#ID_X_]], i32 %[[#ID_Y]], i64 1
// CHECK-DXIL:  call void @{{.*}}bar{{.*}}(<2 x i32> %[[#ID_XY]])
[shader("compute")]
[numthreads(8,8,1)]
void bar(uint2 Idx : SV_DispatchThreadID) {}

