// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV

// SV_InnerCoverage is lowered to the FullyCoveredEXT built-in, which is a
// boolean. It does not consume a location.
// CHECK-SPIRV-DAG: @SV_InnerCoverage = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV-DAG: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]

// CHECK-DXIL-LABEL: define void @main()
// CHECK-SPIRV-LABEL: define void @main()
float4 main(uint ic : SV_InnerCoverage) : SV_Target {
  // The inner coverage is read with an intrinsic, not from the signature.
  // CHECK-DXIL: %[[INPUT:.*]] = call i32 @llvm.dx.inner.coverage()
  // CHECK-DXIL: %[[RESULT:.*]] = call <4 x float> @_Z4mainj(i32 %[[INPUT]])
  // CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %[[RESULT]])

  // The boolean is converted to the uint where only bit 0 may be set.
  // CHECK-SPIRV: %[[LOAD:.*]] = load i1, ptr addrspace(7) @SV_InnerCoverage, align 1
  // CHECK-SPIRV: %[[INPUT:.*]] = zext i1 %[[LOAD]] to i32
  // CHECK-SPIRV: %[[RESULT:.*]] = call spir_func <4 x float> @_Z4mainj(i32 %[[INPUT]])
  // CHECK-SPIRV: store <4 x float> %[[RESULT]], ptr addrspace(8) @SV_Target0, align 4
  return ic ? float4(1, 1, 1, 1) : float4(0, 0, 0, 0);
}

// CHECK-DXIL: declare i32 @llvm.dx.inner.coverage() #[[#ATTR:]]
// CHECK-DXIL: attributes #[[#ATTR]] = { {{.*}}memory(none){{.*}} }

// SV_InnerCoverage is not in the input signature.
// CHECK-DXIL: !dx.semantic.signatures = !{![[#SIG:]]}
// CHECK-DXIL: ![[#SIG]] = !{ptr @main, null, ![[#OUT:]]}
// CHECK-DXIL: ![[#OUT]] = !{![[#TARGET:]]}
// CHECK-DXIL: ![[#TARGET]] = !{i32 0, !"SV_Target", {{.*}}}

// As in DXC, the integer fragment input is flat.
// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]], ![[#FLAT:]]}
// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 5264}
//                                      |       `-> BuiltIn FullyCoveredEXT
//                                      `-> SPIR-V decoration 'BuiltIn'
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV-DAG: ![[#MD_3]] = !{i32 30, i32 0}
