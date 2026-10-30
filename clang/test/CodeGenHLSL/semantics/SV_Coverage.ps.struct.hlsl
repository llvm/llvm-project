// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV

// SV_Coverage is a SPIR-V built-in, so it does not consume a location.
// CHECK-SPIRV-DAG: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#LOC:]]
// CHECK-SPIRV-DAG: @SV_Coverage = external hidden thread_local addrspace(8) global [1 x i32], !spirv.Decorations ![[#BUILTIN:]]

struct PSOut {
  float4 Color : SV_Target;
  uint Mask : SV_Coverage;
};

// CHECK-DXIL-LABEL: define void @main()
// CHECK-SPIRV-LABEL: define void @main()
PSOut main(float4 p : SV_Position) {
  // CHECK-DXIL: %[[POS:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL: call void @_Z4mainDv4_f(ptr %[[OUT:.*]], <4 x float> %[[POS]])
  // CHECK-DXIL: %[[RESULT:.*]] = load %struct.PSOut, ptr %[[OUT]], align 4
  // CHECK-DXIL: %[[COLOR:.*]] = extractvalue %struct.PSOut %[[RESULT]], 0
  // CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %[[COLOR]])
  // CHECK-DXIL: %[[MASK:.*]] = extractvalue %struct.PSOut %[[RESULT]], 1
  // CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0, i32 %[[MASK]])

  // CHECK-SPIRV: %[[RESULT:.*]] = load %struct.PSOut, ptr %{{.*}}, align 4
  // CHECK-SPIRV: %[[COLOR:.*]] = extractvalue %struct.PSOut %[[RESULT]], 0
  // CHECK-SPIRV: store <4 x float> %[[COLOR]], ptr addrspace(8) @SV_Target0, align 4
  // CHECK-SPIRV: %[[MASK:.*]] = extractvalue %struct.PSOut %[[RESULT]], 1
  // CHECK-SPIRV: store i32 %[[MASK]], ptr addrspace(8) @SV_Coverage, align 4
  PSOut o = { p, 3 };
  return o;
}

// SV_Target is allocated to its semantic index, SV_Coverage is not packed.
// CHECK-DXIL: !{{[0-9]+}} = !{![[#TARGET:]], ![[#COV:]]}
// CHECK-DXIL: ![[#TARGET]] = !{i32 0, !"SV_Target", i32 9, i32 16, ![[#IDX:]], i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL: ![[#COV]] = !{i32 1, !"SV_Coverage", i32 5, i32 14, ![[#IDX]], i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}

// CHECK-SPIRV-DAG: ![[#LOC]] = !{![[#LOC_0:]]}
// CHECK-SPIRV-DAG: ![[#LOC_0]] = !{i32 30, i32 0}
// CHECK-SPIRV-DAG: ![[#BUILTIN]] = !{![[#BUILTIN_0:]]}
// CHECK-SPIRV-DAG: ![[#BUILTIN_0]] = !{i32 11, i32 20}
