// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_DispatchThreadID -o - %s | FileCheck %s -DINTRINSIC=dx.thread.id
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_GroupThreadID -o - %s | FileCheck %s -DINTRINSIC=dx.thread.id.in.group
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_GroupID -o - %s | FileCheck %s -DINTRINSIC=dx.group.id
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_DispatchThreadID -o - %s | FileCheck %s -DINTRINSIC=spv.thread.id.i32
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_GroupThreadID -o - %s | FileCheck %s -DINTRINSIC=spv.thread.id.in.group.i32
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -fnative-half-type -fnative-int16-type -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DSEMANTIC=SV_GroupID -o - %s | FileCheck %s -DINTRINSIC=spv.group.id.i32

// Both signed and unsigned 16-bit inputs truncate the i32 intrinsic result.
// CHECK-LABEL: define void @signed_scalar()
// CHECK: %[[RAW:.*]] = call i32 @llvm.[[INTRINSIC]](i32 0)
// CHECK-NEXT: %[[ID:.*]] = trunc i32 %[[RAW]] to i16
// CHECK-NEXT: call {{(spir_func )?}}void @{{.*}}signed_scalar{{.*}}(i16 %[[ID]])
[shader("compute")]
[numthreads(1, 1, 1)]
void signed_scalar(int16_t ID : SEMANTIC) {}

// CHECK-LABEL: define void @unsigned_scalar()
// CHECK: %[[RAW:.*]] = call i32 @llvm.[[INTRINSIC]](i32 0)
// CHECK-NEXT: %[[ID:.*]] = trunc i32 %[[RAW]] to i16
// CHECK-NEXT: call {{(spir_func )?}}void @{{.*}}unsigned_scalar{{.*}}(i16 %[[ID]])
[shader("compute")]
[numthreads(1, 1, 1)]
void unsigned_scalar(uint16_t ID : SEMANTIC) {}

// CHECK-LABEL: define void @vector_one()
// CHECK: %[[RAW:.*]] = call i32 @llvm.[[INTRINSIC]](i32 0)
// CHECK-NEXT: %[[X:.*]] = trunc i32 %[[RAW]] to i16
// CHECK-NEXT: %[[V:.*]] = insertelement <1 x i16> poison, i16 %[[X]], i64 0
// CHECK-NEXT: call {{(spir_func )?}}void @{{.*}}vector_one{{.*}}(<1 x i16> %[[V]])
[shader("compute")]
[numthreads(1, 1, 1)]
void vector_one(int16_t1 ID : SEMANTIC) {}

// CHECK-LABEL: define void @signed_vector()
// CHECK: %[[RAW_X:.*]] = call i32 @llvm.[[INTRINSIC]](i32 0)
// CHECK-NEXT: %[[X:.*]] = trunc i32 %[[RAW_X]] to i16
// CHECK-NEXT: %[[VX:.*]] = insertelement <2 x i16> poison, i16 %[[X]], i64 0
// CHECK-NEXT: %[[RAW_Y:.*]] = call i32 @llvm.[[INTRINSIC]](i32 1)
// CHECK-NEXT: %[[Y:.*]] = trunc i32 %[[RAW_Y]] to i16
// CHECK-NEXT: %[[VXY:.*]] = insertelement <2 x i16> %[[VX]], i16 %[[Y]], i64 1
// CHECK-NEXT: call {{(spir_func )?}}void @{{.*}}signed_vector{{.*}}(<2 x i16> %[[VXY]])
[shader("compute")]
[numthreads(1, 1, 1)]
void signed_vector(int16_t2 ID : SEMANTIC) {}

// CHECK-LABEL: define void @unsigned_vector()
// CHECK: %[[RAW_X:.*]] = call i32 @llvm.[[INTRINSIC]](i32 0)
// CHECK-NEXT: %[[X:.*]] = trunc i32 %[[RAW_X]] to i16
// CHECK-NEXT: %[[VX:.*]] = insertelement <3 x i16> poison, i16 %[[X]], i64 0
// CHECK-NEXT: %[[RAW_Y:.*]] = call i32 @llvm.[[INTRINSIC]](i32 1)
// CHECK-NEXT: %[[Y:.*]] = trunc i32 %[[RAW_Y]] to i16
// CHECK-NEXT: %[[VXY:.*]] = insertelement <3 x i16> %[[VX]], i16 %[[Y]], i64 1
// CHECK-NEXT: %[[RAW_Z:.*]] = call i32 @llvm.[[INTRINSIC]](i32 2)
// CHECK-NEXT: %[[Z:.*]] = trunc i32 %[[RAW_Z]] to i16
// CHECK-NEXT: %[[VXYZ:.*]] = insertelement <3 x i16> %[[VXY]], i16 %[[Z]], i64 2
// CHECK-NEXT: call {{(spir_func )?}}void @{{.*}}unsigned_vector{{.*}}(<3 x i16> %[[VXYZ]])
[shader("compute")]
[numthreads(1, 1, 1)]
void unsigned_vector(uint16_t3 ID : SEMANTIC) {}
