// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-DXIL %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV %s

// CHECK-SPIRV: @SV_StartInstanceLocation = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// The start instance is read with a dedicated operation and is not part of the
// input signature, so the user input that follows keeps signature element 0.

// CHECK-DXIL-LABEL: define void @main()
// CHECK-DXIL: %[[START:.*]] = call i32 @llvm.dx.start.instance.location()
// CHECK-DXIL: %[[POS:.*]] = call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: call <4 x float> @_Z4mainjDv4_f(i32 %[[START]], <4 x float> %[[POS]])
// CHECK-DXIL-NOT: !"SV_StartInstanceLocation"

// CHECK-SPIRV-LABEL: define void @main()
// CHECK-SPIRV: %[[START:.*]] = load i32, ptr addrspace(7) @SV_StartInstanceLocation, align 4
// CHECK-SPIRV: call spir_func <4 x float> @_Z4mainjDv4_f(i32 %[[START]],
float4 main(uint start : SV_StartInstanceLocation, float4 pos : POSITION) : SV_Position {
  return pos + start;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// The metadata below contains SPIR-V decoration 'BuiltIn' followed by
// BuiltIn BaseInstance.
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 4425}
