// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple spirv-unknown-vulkan1.3-library %s  \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// CHECK: @SV_TARGET0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations !0

struct S {
  float4 a : SV_Target;
};


// CHECK-SPIRV: define internal spir_func void @_Z7ps_mainv(ptr dead_on_unwind noalias writable sret(%struct.S) align 1 %agg.result)
// CHECK: %a = getelementptr inbounds nuw %struct.S, ptr %agg.result, i32 0, i32 0
// CHECK: store <4 x float> zeroinitializer, ptr %a, align 1

// CHECK-SPIRV: define void @ps_main()
// CHECK:         %[[#VAR:]] = alloca %struct.S, align 16
// CHECK-SPIRV:                call spir_func void @_Z7ps_mainv(ptr %[[#VAR]])
// CHECK:          %[[#L1:]] = load %struct.S, ptr %[[#VAR]], align 16
// CHECK:          %[[#L2:]] = extractvalue %struct.S %[[#L1]], 0
// CHECK:          store <4 x float> %[[#L2]], ptr addrspace(8) @SV_TARGET0, align 16
[shader("pixel")]
S ps_main() {
  S s;
  s.a = float4(0.f);
  return s;
};
