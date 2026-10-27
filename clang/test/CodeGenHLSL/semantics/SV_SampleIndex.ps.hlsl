// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK-DXIL,DXIL32 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefixes=CHECK-DXIL,DXIL16 %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefix=SPIRV16 %s

#ifdef INT16
#define TYPE int16_t
#else
#define TYPE int
#endif

// CHECK-SPIRV: @SV_SampleIndex = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// SV_SampleIndex is read with a dedicated operation. It still adds an unpacked
// element to the input signature that does not consume a row, so the user
// input that follows it is packed at row 0.

// CHECK-DXIL-LABEL: define void @main()
// CHECK-DXIL: %[[SI:.*]] = call i32 @llvm.dx.sample.index()
// DXIL32-NOT: trunc
// DXIL16: %[[SI16:.*]] = trunc i32 %[[SI]] to i16
// CHECK-DXIL: %[[A:.*]] = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 0, i32 poison)
// DXIL32: call float @_Z4mainif(i32 %[[SI]], float %[[A]])
// DXIL16: call float @_Z4mainsf(i16 %[[SI16]], float %[[A]])

// Signed declarations use an unsigned DXIL component type, with nointerpolation.
// DXIL32-DAG: !{i32 0, !"SV_SampleIndex", i32 5, i32 12, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// DXIL16-DAG: !{i32 0, !"SV_SampleIndex", i32 3, i32 12, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: !{i32 1, !"A", i32 9, i32 0, ![[#ZERO]], i32 2, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: ![[#ZERO]] = !{i32 0}

// CHECK-SPIRV-LABEL: define void @main()
// CHECK-SPIRV: %[[SI:.*]] = load i32, ptr addrspace(7) @SV_SampleIndex, align 4
// CHECK-SPIRV: call spir_func float @_Z4mainif(i32 %[[SI]],
float main(TYPE s : SV_SampleIndex, float a : A) : SV_Target {
  return a * s;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#BUILTIN:]], ![[#FLAT:]]}
// The metadata below contains SPIR-V decoration 'BuiltIn' followed by
// BuiltIn SampleId.
// CHECK-SPIRV-DAG: ![[#BUILTIN]] = !{i32 11, i32 18}
// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}

// The SPIR-V built-in is a 32-bit integer, so a 16-bit declaration is
// truncated.
// SPIRV16: @SV_SampleIndex = {{.*}} addrspace(7) externally_initialized constant i32,
// SPIRV16-LABEL: define void @main()
// SPIRV16: %[[V32:.*]] = load i32, ptr addrspace(7) @SV_SampleIndex
// SPIRV16: %[[V16:.*]] = trunc i32 %[[V32]] to i16
// SPIRV16: call spir_func float @_Z4mainsf(i16 %[[V16]],
