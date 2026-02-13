// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-DX,CHECK
// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-VK,CHECK


struct Input {
  float Idx : SV_Position0;
  float Gid : SV_Position1;
};

struct Output {
  float a : A;
  float b : B;
};

// Make sure SV_DispatchThreadID translated into dx.thread.id.

// CHECK-DX: define hidden void @_Z3foo5Input(ptr dead_on_unwind noalias writable sret(%struct.Output) align 1 %agg.result, ptr noundef byval(%struct.Input) align 1 %input)
// CHECK-VK: define hidden spir_func void @_Z3foo5Input(ptr dead_on_unwind noalias writable sret(%struct.Output) align 1 %agg.result, ptr noundef byval(%struct.Input) align 1 %input)

// CHECK: %Idx = getelementptr inbounds nuw %struct.Input, ptr %input, i32 0, i32 0
// CHECK: %[[#tmp:]] = load float, ptr %Idx, align 1
// CHECK: %a = getelementptr inbounds nuw %struct.Output, ptr %agg.result, i32 0, i32 0
// CHECK: store float %[[#tmp]], ptr %a, align 1
// CHECK: %Gid = getelementptr inbounds nuw %struct.Input, ptr %input, i32 0, i32 1
// CHECK: %[[#tmp:]] = load float, ptr %Gid, align 1
// CHECK: %b = getelementptr inbounds nuw %struct.Output, ptr %agg.result, i32 0, i32 1
// CHECK: store float %[[#tmp]], ptr %b, align 1

Output foo(Input input) {
  Output o;
  o.a = input.Idx;
  o.b = input.Gid;
  return o;
}

