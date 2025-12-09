// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv


struct Inner {
  uint Gid;
};

struct Input {
  uint Idx : SV_DispatchThreadID;
  Inner inner : SV_GroupIndex;
};

// Make sure SV_DispatchThreadID translated into dx.thread.id.

// CHECK:       define void @foo()
// CHECK-DXIL:  %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id(i32 0)
// CHECK-SPIRV: %[[#ID:]] = call i32 @llvm.[[TARGET]].thread.id.i32(i32 0)
// CHECK:     %[[#TMP1:]] = insertvalue %struct.Input poison, i32 %[[#ID]], 0
// CHECK-DXIL: %[[#GID:]] = call i32 @llvm.dx.flattened.thread.id.in.group()
// CHECK-SPIRV:%[[#GID:]] = call i32 @llvm.spv.flattened.thread.id.in.group()
// CHECK:     %[[#TMP2:]] = insertvalue %struct.Inner poison, i32 %[[#GID]], 0
// CHECK:     %[[#TMP3:]] = insertvalue %struct.Input %[[#TMP1]], %struct.Inner %[[#TMP2]], 1
// CHECK:      %[[#VAR:]] = alloca %struct.Input, align 8
// CHECK:                   store %struct.Input %[[#TMP3]], ptr %[[#VAR]], align 4
// CHECK-DXIL:              call void @{{.*}}foo{{.*}}(ptr %[[#VAR]])
// CHECK-SPIRV:             call spir_func void @{{.*}}foo{{.*}}(ptr %[[#VAR]])
[shader("compute")]
[numthreads(8,8,1)]
void foo(Input input) {}
