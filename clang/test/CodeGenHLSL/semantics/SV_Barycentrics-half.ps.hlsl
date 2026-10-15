// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.2-pixel -fnative-half-type -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -fnative-half-type -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,PERSPECTIVE
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -fnative-half-type -finclude-default-header -DAFFINE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,AFFINE

#ifdef AFFINE
#define INTERPOLATION noperspective
#else
#define INTERPOLATION linear
#endif

half4 main(INTERPOLATION half3 barycentrics : SV_Barycentrics) : SV_Target {
  return half4(barycentrics, 1);
}

// DXIL-LABEL: define void @main()
// DXIL: call <3 x half> @llvm.dx.load.input.v3f16(i32 0, i32 0, i8 0, i32 poison)
// DXIL-NOT: @llvm.dx.load.input.v3f32

// The SPIR-V built-in must remain float3, not half3. Only the loaded value
// is truncated to the HLSL input's component type.
// SPIRV: @SV_Barycentrics = external hidden thread_local addrspace(7) externally_initialized constant <3 x float>, !spirv.Decorations ![[BARY:[0-9]+]]
// SPIRV-LABEL: define void @main()
// SPIRV: [[BARY32:%.*]] = load <3 x float>, ptr addrspace(7) @SV_Barycentrics
// SPIRV-NEXT: [[BARY16:%.*]] = fptrunc <3 x float> [[BARY32]] to <3 x half>
// SPIRV: ![[BARY]] = !{![[BUILTIN:[0-9]+]]}
// PERSPECTIVE: ![[BUILTIN]] = !{i32 11, i32 5286}
// AFFINE: ![[BUILTIN]] = !{i32 11, i32 5287}
