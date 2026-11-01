// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefixes=CHECK-DXIL,CHECK-DXIL32 %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefixes=CHECK-DXIL,CHECK-DXIL16 %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -fnative-half-type -fnative-int16-type -disable-llvm-passes -DINT16 -o - %s | FileCheck --check-prefix=CHECK-SPIRV16 %s

#ifdef INT16
#define TYPE int16_t
#else
#define TYPE int
#endif

// CHECK-SPIRV: @SV_PrimitiveID = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// Pixel shaders read SV_PrimitiveID from the input signature as a
// system-generated value, allocated like the other inputs.

// CHECK-DXIL-LABEL: define void @main()
// CHECK-DXIL32: %[[PID:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL16: %[[PID:.*]] = call i16 @llvm.dx.load.input.i16(i32 0, i32 0, i8 0, i32 poison)
// CHECK-DXIL: %[[A:.*]] = call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 0, i32 poison)
// CHECK-DXIL32: call float @_Z4mainif(i32 %[[PID]], float %[[A]])
// CHECK-DXIL16: call float @_Z4mainsf(i16 %[[PID]], float %[[A]])

// Signed declarations use an unsigned DXIL component type, with nointerpolation.
// CHECK-DXIL32-DAG: !{i32 0, !"SV_PrimitiveID", i32 5, i32 10, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL16-DAG: !{i32 0, !"SV_PrimitiveID", i32 3, i32 10, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: !{i32 1, !"A", i32 9, i32 0, ![[#ZERO]], i32 2, i32 1, i8 1, i32 1, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: ![[#ZERO]] = !{i32 0}

// CHECK-SPIRV-LABEL: define void @main()
// CHECK-SPIRV: %[[PID:.*]] = load i32, ptr addrspace(7) @SV_PrimitiveID, align 4
// CHECK-SPIRV: call spir_func float @_Z4mainif(i32 %[[PID]],
float main(TYPE pid : SV_PrimitiveID, float a : A) : SV_Target {
  return a * pid;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#BUILTIN:]], ![[#FLAT:]]}
// The metadata below contains SPIR-V decoration 'BuiltIn' followed by
// BuiltIn PrimitiveId.
// CHECK-SPIRV-DAG: ![[#BUILTIN]] = !{i32 11, i32 7}
// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}

// The SPIR-V built-in is a 32-bit integer, so a 16-bit declaration is
// truncated.
// CHECK-SPIRV16: @SV_PrimitiveID = {{.*}} addrspace(7) externally_initialized constant i32,
// CHECK-SPIRV16-LABEL: define void @main()
// CHECK-SPIRV16: %[[V32:.*]] = load i32, ptr addrspace(7) @SV_PrimitiveID
// CHECK-SPIRV16: %[[V16:.*]] = trunc i32 %[[V32]] to i16
// CHECK-SPIRV16: call spir_func float @_Z4mainsf(i16 %[[V16]],
