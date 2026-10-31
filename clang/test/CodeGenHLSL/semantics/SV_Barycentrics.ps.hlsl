// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,AFFINE-CENTER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -DPERSPECTIVE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,PERSPECTIVE-CENTER
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -DCENTROID -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,AFFINE-CENTROID
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -DPERSPECTIVE -DCENTROID -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,PERSPECTIVE-CENTROID
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -DSAMPLE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,AFFINE-SAMPLE
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-pixel -finclude-default-header -DPERSPECTIVE -DSAMPLE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=DXIL,PERSPECTIVE-SAMPLE
// SPIR-V builtin selection and sampling decorations are independent.
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,SPIRV-AFFINE
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -DPERSPECTIVE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,SPIRV-PERSPECTIVE
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -DCENTROID -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,SPIRV-AFFINE,SPIRV-CENTROID
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -finclude-default-header -DPERSPECTIVE -DSAMPLE -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=SPIRV,SPIRV-PERSPECTIVE,SPIRV-SAMPLE

#ifdef CENTROID
#define SAMPLING centroid
#elif defined(SAMPLE)
#define SAMPLING sample
#else
#define SAMPLING
#endif
#ifdef PERSPECTIVE
#define INTERPOLATION linear
#elif defined(CENTROID) || defined(SAMPLE)
#define INTERPOLATION noperspective
#else
#define INTERPOLATION
#endif

struct Inputs {
  SAMPLING INTERPOLATION float3 barycentrics : SV_Barycentrics0;
  float value : VALUE;
};

// The parameter's noperspective is inherited unless the field overrides it.
// Each entry has exactly one barycentric input, which consumes no packed row
// or SPIR-V location.
float4 main(noperspective Inputs input) : SV_Target {
  return float4(input.barycentrics, input.value);
}

// DXIL-LABEL: define void @main()
// DXIL: call <3 x float> @llvm.dx.load.input.v3f32(i32 0, i32 0, i8 0, i32 poison)
// DXIL: call float @llvm.dx.load.input.f32(i32 1, i32 0, i8 0, i32 poison)
// AFFINE-CENTER-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 4, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// PERSPECTIVE-CENTER-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 2, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// AFFINE-CENTROID-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 5, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// PERSPECTIVE-CENTROID-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 3, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// AFFINE-SAMPLE-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 7, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// PERSPECTIVE-SAMPLE-DAG: !{i32 0, !"SV_Barycentrics", i32 9, i32 28, !{{[0-9]+}}, i32 6, i32 1, i8 3, i32 -1, i8 -1, i8 0, i8 0, i32 0}
// DXIL-DAG: !{i32 1, !"VALUE", i32 9, i32 0, ![[ZERO:[0-9]+]], i32 4, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// DXIL-DAG: ![[ZERO]] = !{i32 0}

// SPIRV-DAG: @SV_Barycentrics = external hidden thread_local addrspace(7) externally_initialized constant <3 x float>, !spirv.Decorations ![[BARY:[0-9]+]]
// SPIRV-DAG: @VALUE0 = {{.*}} !spirv.Decorations ![[VALUE:[0-9]+]]
// SPIRV-LABEL: define void @main()
// SPIRV: load <3 x float>, ptr addrspace(7) @SV_Barycentrics
// SPIRV-NOT: fptrunc
// SPIRV-DAG: ![[BARY]] = !{![[BUILTIN:[0-9]+]]{{.*}}}
// SPIRV-DAG: ![[VALUE]] = !{![[VALUE_LOCATION:[0-9]+]], ![[NOPERSP:[0-9]+]]}
// SPIRV-AFFINE-DAG: ![[BUILTIN]] = !{i32 11, i32 5287}
// SPIRV-PERSPECTIVE-DAG: ![[BUILTIN]] = !{i32 11, i32 5286}
// SPIRV-DAG: ![[VALUE_LOCATION]] = !{i32 30, i32 0}
// SPIRV-DAG: ![[NOPERSP]] = !{i32 13}
// SPIRV-CENTROID-DAG: !{i32 16}
// SPIRV-SAMPLE-DAG: !{i32 17}
