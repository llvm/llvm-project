// RUN: %clang_cc1 -triple spirv-unknown-vulkan-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-DXIL -DTARGET=dx

// CHECK-SPIRV-DAG:  @AAA0 = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations ![[#METADATA_0:]]
// CHECK-SPIRV-DAG:    @B0 = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#METADATA_2:]]
// CHECK-SPIRV-DAG:   @CC0 = external hidden thread_local addrspace(7) externally_initialized constant <2 x float>, !spirv.Decorations ![[#METADATA_4:]]


// FIXME: replace `float2 c` with a  matrix when available.
void main(float a : AAA, int b : B, float2 c : CC) {
  float tmp = a + b + c.x + c.y;
}
// CHECK-SPIRV: define internal spir_func void @_Z4mainfiDv2_f(float noundef nofpclass(nan inf) %a, i32 noundef %b, <2 x float> noundef nofpclass(nan inf) %c) #0 {

// CHECK: define void @main()

// CHECK-DXIL: %AAA0 = call float @llvm.dx.load.input.f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL:   %B0 = call i32 @llvm.dx.load.input.i32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL   %CC0 = call <2 x float> @llvm.dx.load.input.v2f32(i32 4, i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL:         call void @_Z4mainfiDv2_f(float %AAA0, i32 %B0, <2 x float> %CC0)

// CHECK-SPIRV: %[[#AAA0:]] = load float, ptr addrspace(7) @AAA0, align 4
// CHECK-SPIRV:   %[[#B0:]] = load i32, ptr addrspace(7) @B0, align 4
// CHECK-SPIRV:  %[[#CC0:]] = load <2 x float>, ptr addrspace(7) @CC0, align 8
// CHECK-SPIRV:               call spir_func void @_Z4mainfiDv2_f(float %[[#AAA0]], i32 %[[#B0]], <2 x float> %[[#CC0]]) [ "convergencectrl"(token %0) ]


// CHECK-SPIRV-DAG: ![[#METADATA_0]] = !{![[#METADATA_1:]]}
// CHECK-SPIRV-DAG: ![[#METADATA_2]] = !{![[#METADATA_3:]]}
// CHECK-SPIRV-DAG: ![[#METADATA_4]] = !{![[#METADATA_5:]]}

// CHECK-SPIRV-DAG: ![[#METADATA_1]] = !{i32 30, i32 0}
// CHECK-SPIRV-DAG: ![[#METADATA_3]] = !{i32 30, i32 1}
// CHECK-SPIRV-DAG: ![[#METADATA_5]] = !{i32 30, i32 2}
//                                            |      `- Location index
//                                            `-> Decoration "Location"
