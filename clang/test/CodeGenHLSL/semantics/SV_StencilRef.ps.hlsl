// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK-DXIL,DXIL32 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefixes=CHECK-DXIL,DXIL16 %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefix=SPIRV16 %s

#ifdef INT16
#define TYPE int16_t
#else
#define TYPE int
#endif

struct Output {
  TYPE Stencil : SV_StencilRef;
  float4 Color : SV_Target;
};

Output main(float4 p : SV_Position) {
  Output O;
  O.Stencil = 3;
  O.Color = p;
  return O;
}

// The stencil reference is an output element that does not consume a render
// target row.
// CHECK-DXIL-LABEL: define void @main()
// DXIL32: call void @llvm.dx.store.output.i32(i32 0, i32 0, i8 0, i32 %{{.*}})
// DXIL16: call void @llvm.dx.store.output.i16(i32 0, i32 0, i8 0, i16 %{{.*}})
// CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 1, i32 0, i8 0, <4 x float> %{{.*}})

// Signed declarations use an unsigned DXIL component type.
// DXIL32-DAG: !{i32 0, !"SV_StencilRef", i32 5, i32 20, ![[#ZERO:]], i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// DXIL16-DAG: !{i32 0, !"SV_StencilRef", i32 3, i32 20, ![[#ZERO:]], i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: !{i32 1, !"SV_Target", i32 9, i32 16, ![[#ZERO]], i32 0, i32 1, i8 4, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: ![[#ZERO]] = !{i32 0}

// CHECK-SPIRV-DAG: @SV_StencilRef = external hidden thread_local addrspace(8) global i32, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV-DAG: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]
// CHECK-SPIRV-LABEL: define void @main()
// CHECK-SPIRV: store i32 %{{.*}}, ptr addrspace(8) @SV_StencilRef, align 4
// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#BUILTIN:]]}
// CHECK-SPIRV-DAG: ![[#BUILTIN]] = !{i32 11, i32 5014}
//                                      |       `-> BuiltIn FragStencilRefEXT
//                                      `-> SPIR-V decoration 'BuiltIn'
// The render target keeps location 0.
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{![[#LOC:]]}
// CHECK-SPIRV-DAG: ![[#LOC]] = !{i32 30, i32 0}

// The SPIR-V built-in is a 32-bit integer, so a signed 16-bit declaration is
// sign extended.
// SPIRV16: @SV_StencilRef = {{.*}} addrspace(8) global i32,
// SPIRV16-LABEL: define void @main()
// SPIRV16: %[[S32:.*]] = sext i16 %{{.*}} to i32
// SPIRV16: store i32 %[[S32]], ptr addrspace(8) @SV_StencilRef
