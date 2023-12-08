// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s

// Make sure SV_DispatchThreadID translated into dx.thread.id.

const RWBuffer<float> In;
RWBuffer<float> Out;

// CHECK: define void @foo()
// CHECK: %[[ID:[0-9a-zA-Z]+]] = call i32 @llvm.dx.thread.id(i32 0)
// CHECK: call void @"?foo@@YAXH@Z"(i32 %[[ID]])
[shader("compute")]
[numthreads(8,8,1)]
void foo(uint Idx : SV_DispatchThreadID) {
  Out[Idx] = In[Idx];
}

// CHECK: define void @bar()
// CHECK: %[[ID_X:[0-9a-zA-Z]+]] = call i32 @llvm.dx.thread.id(i32 0)
// CHECK: %[[ID_X_:[0-9a-zA-Z]+]] = insertelement <2 x i32> poison, i32 %[[ID_X]], i64 0
// CHECK: %[[ID_Y:[0-9a-zA-Z]+]] = call i32 @llvm.dx.thread.id(i32 1)
// CHECK: %[[ID_XY:[0-9a-zA-Z]+]] = insertelement <2 x i32> %[[ID_X_]], i32 %[[ID_Y]], i64 1
// CHECK: call void @"?bar@@YAXT?$__vector@H$01@__clang@@@Z"(<2 x i32> %[[ID_XY]])
[shader("compute")]
[numthreads(8,8,1)]
void bar(uint2 Idx : SV_DispatchThreadID) {
  Out[Idx.y] = In[Idx.x];
}

