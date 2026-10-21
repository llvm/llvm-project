// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck %s --check-prefix=SPIRV16

#ifdef INT16
#define OUT_TYPE int16_t
#define IN_TYPE uint16_t
#else
#define OUT_TYPE int
#define IN_TYPE uint
#endif

struct VSOut {
  float4 Pos : SV_Position;
  OUT_TYPE Viewport : SV_ViewportArrayIndex;
};

// The vertex shader writes the viewport index as a system value. Its own
// input with the same name is an arbitrary semantic.
[shader("vertex")]
VSOut vs_main(uint In : SV_ViewportArrayIndex) {
  VSOut O;
  O.Pos = 0;
  O.Viewport = In + 1;
  return O;
}

[shader("pixel")]
float4 ps_main(float4 Pos : SV_Position, IN_TYPE Viewport : SV_ViewportArrayIndex)
    : SV_Target {
  return Pos * Viewport;
}

// CHECK-DXIL-LABEL: define void @vs_main()
// CHECK-DXIL: call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0,
// CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 1, i32 0, i8 0,
// CHECK-DXIL-LABEL: define void @ps_main()
// CHECK-DXIL: call <4 x float> @llvm.dx.load.input.v4f32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)

// The vertex input is arbitrary (kind 0). The output and the pixel input are
// unsigned (5), nointerpolation (1) ViewportArrayIndex (kind 5) elements.
// CHECK-DXIL-DAG: !{ptr @vs_main, ![[#VS_IN:]], ![[#VS_OUT:]]}
// CHECK-DXIL-DAG: ![[#VS_IN]] = !{![[#VS_IN_VP:]]}
// CHECK-DXIL-DAG: ![[#VS_IN_VP]] = !{i32 0, !"SV_ViewportArrayIndex", i32 5, i32 0, ![[#ZERO:]], i32 0, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: ![[#VS_OUT]] = !{![[#VS_OUT_POS:]], ![[#VS_OUT_VP:]]}
// CHECK-DXIL-DAG: ![[#VS_OUT_VP]] = !{i32 1, !"SV_ViewportArrayIndex", i32 5, i32 5, ![[#ZERO]], i32 1, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
// The pixel input signature is identical to the vertex output signature.
// CHECK-DXIL-DAG: !{ptr @ps_main, ![[#VS_OUT]], ![[#PS_OUT:]]}
// CHECK-DXIL-DAG: ![[#ZERO]] = !{i32 0}

// CHECK-SPIRV-DAG: @SV_ViewportArrayIndex0 = {{.*}} addrspace(7) {{.*}} i32, !spirv.Decorations ![[#VS_IN_LOC:]]
// CHECK-SPIRV-DAG: @SV_ViewportArrayIndex = {{.*}} addrspace(8) global i32, !spirv.Decorations ![[#VS_OUT_BUILTIN:]]
// The pixel input has the same flat ViewportIndex decorations as the output.
// CHECK-SPIRV-DAG: @SV_ViewportArrayIndex.{{[0-9]+}} = {{.*}} addrspace(7) {{.*}} i32, !spirv.Decorations ![[#VS_OUT_BUILTIN]]
// CHECK-SPIRV-DAG: ![[#VS_OUT_BUILTIN]] = !{![[#VIEWPORT:]], ![[#FLAT:]]}
// CHECK-SPIRV-DAG: ![[#VIEWPORT]] = !{i32 11, i32 10}
//                                        |       `-> BuiltIn ViewportIndex
//                                        `-> SPIR-V decoration 'BuiltIn'
// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}

// The SPIR-V built-in is a 32-bit integer, so a signed 16-bit output is sign
// extended and a 16-bit input is truncated.
// SPIRV16-LABEL: define void @vs_main()
// SPIRV16: %[[OUT:.*]] = sext i16 %{{.*}} to i32
// SPIRV16: store i32 %[[OUT]], ptr addrspace(8) @SV_ViewportArrayIndex
// SPIRV16-LABEL: define void @ps_main()
// SPIRV16: %[[IN:.*]] = load i32, ptr addrspace(7) @SV_ViewportArrayIndex.{{[0-9]+}}
// SPIRV16: trunc i32 %[[IN]] to i16
