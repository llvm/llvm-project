// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -DTEST_DXIL \
// RUN:   -fmatrix-memory-layout=row-major -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,DXIL_CHECK -DTARGET=dx
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -DTEST_SPIRV \
// RUN:   -fmatrix-memory-layout=row-major -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPIRV_CHECK -DTARGET=spv
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -DTEST_SPIRV_HALF -fnative-half-type \
// RUN:   -fmatrix-memory-layout=row-major -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefix=SPIRV_HALF_CHECK

// CHECK-LABEL: define {{.*}} double @{{.*}}fma_double{{.*}}(
// CHECK: %[[P0:.*]] = load double, ptr %{{.*}}, align 8
// CHECK: %[[P1:.*]] = load double, ptr %{{.*}}, align 8
// CHECK: %[[P2:.*]] = load double, ptr %{{.*}}, align 8
// CHECK: %{{dx|spv}}.fma = call reassoc nnan ninf nsz arcp afn double @llvm.[[TARGET]].fma.f64(double %[[P0]], double %[[P1]], double %[[P2]])
// CHECK: ret double %{{dx|spv}}.fma
double dxil_fma_double(double a, double b, double c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <2 x double> @{{.*}}fma_double2{{.*}}(
// CHECK: %[[P0:.*]] = load <2 x double>, ptr %{{.*}}, align 16
// CHECK: %[[P1:.*]] = load <2 x double>, ptr %{{.*}}, align 16
// CHECK: %[[P2:.*]] = load <2 x double>, ptr %{{.*}}, align 16
// CHECK: %{{dx|spv}}.fma = call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.[[TARGET]].fma.v2f64(<2 x double> %[[P0]], <2 x double> %[[P1]], <2 x double> %[[P2]])
// CHECK: ret <2 x double> %{{dx|spv}}.fma
double2 dxil_fma_double2(double2 a, double2 b, double2 c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <3 x double> @{{.*}}fma_double3{{.*}}(
// CHECK: %[[P0:.*]] = load <3 x double>, ptr %{{.*}}, align 32
// CHECK: %[[P1:.*]] = load <3 x double>, ptr %{{.*}}, align 32
// CHECK: %[[P2:.*]] = load <3 x double>, ptr %{{.*}}, align 32
// CHECK: %{{dx|spv}}.fma = call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.[[TARGET]].fma.v3f64(<3 x double> %[[P0]], <3 x double> %[[P1]], <3 x double> %[[P2]])
// CHECK: ret <3 x double> %{{dx|spv}}.fma
double3 dxil_fma_double3(double3 a, double3 b, double3 c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}fma_double4{{.*}}(
// CHECK: %[[P0:.*]] = load <4 x double>, ptr %{{.*}}, align 32
// CHECK: %[[P1:.*]] = load <4 x double>, ptr %{{.*}}, align 32
// CHECK: %[[P2:.*]] = load <4 x double>, ptr %{{.*}}, align 32
// CHECK: %{{dx|spv}}.fma = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.[[TARGET]].fma.v4f64(<4 x double> %[[P0]], <4 x double> %[[P1]], <4 x double> %[[P2]])
// CHECK: ret <4 x double> %{{dx|spv}}.fma
double4 dxil_fma_double4(double4 a, double4 b, double4 c) { return fma(a, b, c); }

#ifdef TEST_DXIL

// DXIL_CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}dxil_fma_double1x4{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.dx.fma.v4f64(
// DXIL_CHECK: ret <4 x double> %dx.fma
double1x4 dxil_fma_double1x4(double1x4 a, double1x4 b, double1x4 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}dxil_fma_double4x1{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.dx.fma.v4f64(
// DXIL_CHECK: ret <4 x double> %dx.fma
double4x1 dxil_fma_double4x1(double4x1 a, double4x1 b, double4x1 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}dxil_fma_double2x2{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.dx.fma.v4f64(
// DXIL_CHECK: ret <4 x double> %dx.fma
double2x2 dxil_fma_double2x2(double2x2 a, double2x2 b, double2x2 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <6 x double> @{{.*}}dxil_fma_double2x3{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.dx.fma.v6f64(
// DXIL_CHECK: ret <6 x double> %dx.fma
double2x3 dxil_fma_double2x3(double2x3 a, double2x3 b, double2x3 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <6 x double> @{{.*}}dxil_fma_double3x2{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.dx.fma.v6f64(
// DXIL_CHECK: ret <6 x double> %dx.fma
double3x2 dxil_fma_double3x2(double3x2 a, double3x2 b, double3x2 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <9 x double> @{{.*}}dxil_fma_double3x3{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <9 x double> @llvm.dx.fma.v9f64(
// DXIL_CHECK: ret <9 x double> %dx.fma
double3x3 dxil_fma_double3x3(double3x3 a, double3x3 b, double3x3 c) { return fma(a, b, c); }

// DXIL_CHECK-LABEL: define {{.*}} <16 x double> @{{.*}}dxil_fma_double4x4{{.*}}(
// DXIL_CHECK: %dx.fma = call reassoc nnan ninf nsz arcp afn <16 x double> @llvm.dx.fma.v16f64(
// DXIL_CHECK: ret <16 x double> %dx.fma
double4x4 dxil_fma_double4x4(double4x4 a, double4x4 b, double4x4 c) { return fma(a, b, c); }
#endif

#ifdef TEST_SPIRV
// SPIRV_CHECK-LABEL: define {{.*}} float @{{.*}}spv_fma_float{{.*}}(
// SPIRV_CHECK: %[[P0:.*]] = load float, ptr %{{.*}}, align 4
// SPIRV_CHECK: %[[P1:.*]] = load float, ptr %{{.*}}, align 4
// SPIRV_CHECK: %[[P2:.*]] = load float, ptr %{{.*}}, align 4
// SPIRV_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn float @llvm.spv.fma.f32(float %[[P0]], float %[[P1]], float %[[P2]])
// SPIRV_CHECK: ret float %spv.fma
float spv_fma_float(float a, float b, float c) { return fma(a, b, c); }

// SPIRV_CHECK-LABEL: define {{.*}} <2 x float> @{{.*}}spv_fma_float2{{.*}}(
// SPIRV_CHECK: %[[P0:.*]] = load <2 x float>, ptr %{{.*}}, align 8
// SPIRV_CHECK: %[[P1:.*]] = load <2 x float>, ptr %{{.*}}, align 8
// SPIRV_CHECK: %[[P2:.*]] = load <2 x float>, ptr %{{.*}}, align 8
// SPIRV_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.fma.v2f32(<2 x float> %[[P0]], <2 x float> %[[P1]], <2 x float> %[[P2]])
// SPIRV_CHECK: ret <2 x float> %spv.fma
float2 spv_fma_float2(float2 a, float2 b, float2 c) { return fma(a, b, c); }

// SPIRV_CHECK-LABEL: define {{.*}} <3 x float> @{{.*}}spv_fma_float3{{.*}}(
// SPIRV_CHECK: %[[P0:.*]] = load <3 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %[[P1:.*]] = load <3 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %[[P2:.*]] = load <3 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <3 x float> @llvm.spv.fma.v3f32(<3 x float> %[[P0]], <3 x float> %[[P1]], <3 x float> %[[P2]])
// SPIRV_CHECK: ret <3 x float> %spv.fma
float3 spv_fma_float3(float3 a, float3 b, float3 c) { return fma(a, b, c); }

// SPIRV_CHECK-LABEL: define {{.*}} <4 x float> @{{.*}}spv_fma_float4{{.*}}(
// SPIRV_CHECK: %[[P0:.*]] = load <4 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %[[P1:.*]] = load <4 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %[[P2:.*]] = load <4 x float>, ptr %{{.*}}, align 16
// SPIRV_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <4 x float> @llvm.spv.fma.v4f32(<4 x float> %[[P0]], <4 x float> %[[P1]], <4 x float> %[[P2]])
// SPIRV_CHECK: ret <4 x float> %spv.fma
float4 spv_fma_float4(float4 a, float4 b, float4 c) { return fma(a, b, c); }

#endif

#ifdef TEST_SPIRV_HALF
// SPIRV_HALF_CHECK-LABEL: define {{.*}} half @{{.*}}spv_fma_half{{.*}}(
// SPIRV_HALF_CHECK: %[[P0:.*]] = load half, ptr %{{.*}}, align 2
// SPIRV_HALF_CHECK: %[[P1:.*]] = load half, ptr %{{.*}}, align 2
// SPIRV_HALF_CHECK: %[[P2:.*]] = load half, ptr %{{.*}}, align 2
// SPIRV_HALF_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fma.f16(half %[[P0]], half %[[P1]], half %[[P2]])
// SPIRV_HALF_CHECK: ret half %spv.fma
half spv_fma_half(half a, half b, half c) { return fma(a, b, c); }

// SPIRV_HALF_CHECK-LABEL: define {{.*}} <2 x half> @{{.*}}spv_fma_half2{{.*}}(
// SPIRV_HALF_CHECK: %[[P0:.*]] = load <2 x half>, ptr %{{.*}}, align 4
// SPIRV_HALF_CHECK: %[[P1:.*]] = load <2 x half>, ptr %{{.*}}, align 4
// SPIRV_HALF_CHECK: %[[P2:.*]] = load <2 x half>, ptr %{{.*}}, align 4
// SPIRV_HALF_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <2 x half> @llvm.spv.fma.v2f16(<2 x half> %[[P0]], <2 x half> %[[P1]], <2 x half> %[[P2]])
// SPIRV_HALF_CHECK: ret <2 x half> %spv.fma
half2 spv_fma_half2(half2 a, half2 b, half2 c) { return fma(a, b, c); }

// SPIRV_HALF_CHECK-LABEL: define {{.*}} <3 x half> @{{.*}}spv_fma_half3{{.*}}(
// SPIRV_HALF_CHECK: %[[P0:.*]] = load <3 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %[[P1:.*]] = load <3 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %[[P2:.*]] = load <3 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <3 x half> @llvm.spv.fma.v3f16(<3 x half> %[[P0]], <3 x half> %[[P1]], <3 x half> %[[P2]])
// SPIRV_HALF_CHECK: ret <3 x half> %spv.fma
half3 spv_fma_half3(half3 a, half3 b, half3 c) { return fma(a, b, c); }

// SPIRV_HALF_CHECK-LABEL: define {{.*}} <4 x half> @{{.*}}spv_fma_half4{{.*}}(
// SPIRV_HALF_CHECK: %[[P0:.*]] = load <4 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %[[P1:.*]] = load <4 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %[[P2:.*]] = load <4 x half>, ptr %{{.*}}, align 8
// SPIRV_HALF_CHECK: %spv.fma = call reassoc nnan ninf nsz arcp afn <4 x half> @llvm.spv.fma.v4f16(<4 x half> %[[P0]], <4 x half> %[[P1]], <4 x half> %[[P2]])
// SPIRV_HALF_CHECK: ret <4 x half> %spv.fma
half4 spv_fma_half4(half4 a, half4 b, half4 c) { return fma(a, b, c); }
#endif
