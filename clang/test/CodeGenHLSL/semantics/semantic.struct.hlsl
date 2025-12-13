// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv

struct S0 {
  uint Idx : SV_DispatchThreadID;
};

// CHECK:       define void @main0()
// CHECK-DXIL:    %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK-SPIRV:   %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id.i32(i32 0)
// CHECK:        %[[#TMP:]] = insertvalue %struct.S0 poison, i32 %[[#ID:]], 0
// CHECK:        %[[#ARG:]] = alloca %struct.S0, align 8
// CHECK:                     store %struct.S0 %[[#TMP]], ptr %[[#ARG]], align 4
// CHECK-DXIL:                call void @{{.*}}main0{{.*}}(ptr %[[#ARG]])
// CHECK-SPIRV:               call spir_func void @{{.*}}main0{{.*}}(ptr %[[#ARG]])
[shader("compute")]
[numthreads(8,8,1)]
void main0(S0 p) {}

struct S1 {
  uint2  a : SV_DispatchThreadID;
  uint2  b : SV_GroupThreadID;
};

// CHECK:                     define void @main1()
// CHECK-DXIL:    %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK-SPIRV:   %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id.i32(i32 0)
// CHECK:        %[[#AX_:]] = insertelement <2 x i32> poison, i32 %[[#ID]], i64 0
// CHECK-DXIL:    %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 1)
// CHECK-SPIRV:   %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id.i32(i32 1)
// CHECK:        %[[#AXY:]] = insertelement <2 x i32> %[[#AX_]], i32 %[[#ID]], i64 1
// CHECK:       %[[#S1A_:]] = insertvalue %struct.S1 poison, <2 x i32> %[[#AXY]], 0
// CHECK-DXIL:  %[[#ID_X:]] = call i32 @llvm.[[TARGET]].thread.id.in.group(i32 0)
// CHECK-SPIRV: %[[#ID_X:]] = call i32 @llvm.[[TARGET]].thread.id.in.group.i32(i32 0)
// CHECK:      %[[#ID_X_:]] = insertelement <2 x i32> poison, i32 %[[#ID_X]], i64 0
// CHECK-DXIL:  %[[#ID_Y:]] = call i32 @llvm.[[TARGET]].thread.id.in.group(i32 1)
// CHECK-SPIRV: %[[#ID_Y:]] = call i32 @llvm.[[TARGET]].thread.id.in.group.i32(i32 1)
// CHECK:      %[[#ID_XY:]] = insertelement <2 x i32> %[[#ID_X_]], i32 %[[#ID_Y]], i64 1
// CHECK:       %[[#S1AB:]] = insertvalue %struct.S1 %[[#S1A_]], <2 x i32> %[[#ID_XYZ:]], 1
// CHECK:        %[[#ARG:]] = alloca %struct.S1, align 8
// CHECK:                     store %struct.S1 %[[#S1AB]], ptr %[[#ARG]], align 8
// CHECK-DXIL:                call void @{{.*}}main1{{.*}}(ptr %[[#ARG]])
// CHECK-SPIRV:               call spir_func void @{{.*}}main1{{.*}}(ptr %[[#ARG]])
[shader("compute")]
[numthreads(8,8,1)]
void main1(S1 p) {}

struct S2C {
  uint2 b : SV_GroupThreadID;
};

struct S2 {
  uint  a : SV_DispatchThreadID;
  S2C child;
};

// CHECK:                     define void @main2()
// CHECK-DXIL:    %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK-SPIRV:   %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id.i32(i32 0)
// CHECK:       %[[#S2A_:]] = insertvalue %struct.S2 poison, i32 %[[#ID:]], 0

// CHECK-DXIL:  %[[#ID_X:]] = call i32 @llvm.[[TARGET]].thread.id.in.group(i32 0)
// CHECK-SPIRV: %[[#ID_X:]] = call i32 @llvm.[[TARGET]].thread.id.in.group.i32(i32 0)
// CHECK:      %[[#ID_X_:]] = insertelement <2 x i32> poison, i32 %[[#ID_X]], i64 0
// CHECK-DXIL:  %[[#ID_Y:]] = call i32 @llvm.[[TARGET]].thread.id.in.group(i32 1)
// CHECK-SPIRV: %[[#ID_Y:]] = call i32 @llvm.[[TARGET]].thread.id.in.group.i32(i32 1)
// CHECK:      %[[#ID_XY:]] = insertelement <2 x i32> %[[#ID_X_]], i32 %[[#ID_Y]], i64 1
// CHECK:        %[[#S2C:]] = insertvalue %struct.S2C poison, <2 x i32> %[[#ID_XY:]], 0

// CHECK:       %[[#S2AB:]] = insertvalue %struct.S2 %[[#S2A_]], %struct.S2C %[[#S2V:]], 1
// CHECK:        %[[#ARG:]] = alloca %struct.S2, align 8
// CHECK:                     store %struct.S2 %[[#S2AB]], ptr %[[#ARG]], align 1
// CHECK-DXIL:                call void @{{.*}}main2{{.*}}(ptr %[[#ARG]])
// CHECK-SPIRV:               call spir_func void @{{.*}}main2{{.*}}(ptr %[[#ARG]])
[shader("compute")]
[numthreads(8,8,1)]
void main2(S2 p) {}
