// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown | FileCheck --check-prefix=CHECK --check-prefix=NODIVOPT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple spir-unknown-unknown -cl-fp32-correctly-rounded-divide-sqrt | FileCheck --check-prefix=CHECK --check-prefix=DIVOPT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -DNOFP16 -DNOFP64 -cl-std=CL1.2 -triple r600-unknown-unknown -target-cpu r600 -pedantic | FileCheck --check-prefix=CHECK-FLT %s
// RUN: %clang_cc1 %s -emit-llvm -o - -DFP64 -cl-std=CL1.2 -triple spir-unknown-unknown -pedantic | FileCheck --check-prefix=CHECK-DBL %s

#ifndef NOFP16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef __attribute__(( ext_vector_type(4) )) half half4;

half hpscalardiv(half a, half b) {
  // CHECK: @hpscalardiv
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD_HFDIV:[0-9]+]]
  // DIVOPT-NOT: !fpmath !{{[0-9]+}}
  return a / b;
}

half4 hpvectordiv(half4 a, half4 b) {
  // CHECK: @hpvectordiv
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD_HFDIV]]
  // DIVOPT-NOT: !fpmath !{{[0-9]+}}
  return a / b;
}

half elementwise_sqrt_f16(half a) {
  // CHECK-LABEL: @elementwise_sqrt_f16
  // NODIVOPT: call half @llvm.sqrt.f16(half %{{.+}}), !fpmath ![[MD_HSQRT:[0-9]+]]
  // DIVOPT: call half @llvm.sqrt.f16(half %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}

half4 elementwise_sqrt_v4f16(half4 a) {
  // CHECK-LABEL: @elementwise_sqrt_v4f16
  // NODIVOPT: call <4 x half> @llvm.sqrt.v4f16(<4 x half> %{{.+}}), !fpmath ![[MD_HSQRT]]
  // DIVOPT: call <4 x half> @llvm.sqrt.v4f16(<4 x half> %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}

#endif // NOFP16

typedef __attribute__(( ext_vector_type(4) )) float float4;

float spscalardiv(float a, float b) {
  // CHECK: @spscalardiv
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD_FDIV:[0-9]+]]
  // DIVOPT-NOT: !fpmath !{{[0-9]+}}
  return a / b;
}

float4 spvectordiv(float4 a, float4 b) {
  // CHECK: @spvectordiv
  // CHECK: fdiv{{.*}},
  // NODIVOPT: !fpmath ![[MD_FDIV]]
  // DIVOPT-NOT: !fpmath !{{[0-9]+}}
  return a / b;
}

float spscalarsqrt(float a) {
  // CHECK-LABEL: @spscalarsqrt
  // NODIVOPT: call float @llvm.sqrt.f32(float %{{.+}}), !fpmath ![[MD_SQRT:[0-9]+]]
  // DIVOPT: call float @llvm.sqrt.f32(float %{{.+}}){{$}}
  return __builtin_sqrtf(a);
}

float elementwise_sqrt_f32(float a) {
  // CHECK-LABEL: @elementwise_sqrt_f32
  // NODIVOPT: call float @llvm.sqrt.f32(float %{{.+}}), !fpmath ![[MD_SQRT]]
  // DIVOPT: call float @llvm.sqrt.f32(float %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}

float4 elementwise_sqrt_v4f32(float4 a) {
  // CHECK-LABEL: @elementwise_sqrt_v4f32
  // NODIVOPT: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}), !fpmath ![[MD_SQRT]]
  // DIVOPT: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}


#if __OPENCL_C_VERSION__ >=120
void printf(constant char* fmt, ...);

void testdbllit(long *val) {
  // CHECK-FLT: float noundef 2.000000e+01
  // CHECK-DBL: double noundef 2.000000e+01
  printf("%f", 20.0);
}

#endif

#ifndef NOFP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
typedef __attribute__(( ext_vector_type(4) )) double double4;

double dpscalardiv(double a, double b) {
  // CHECK: @dpscalardiv
  // CHECK-NOT: !fpmath
  return a / b;
}

double4 dpvectordiv(double4 a, double4 b) {
  // CHECK: @dpvectordiv
  // CHECK-NOT: !fpmath
  return a / b;
}

double dpscalarsqrt(double a) {
  // CHECK-LABEL: @dpscalarsqrt
  // CHECK: call double @llvm.sqrt.f64(double %{{.+}}){{$}}
  return __builtin_sqrt(a);
}

double elementwise_sqrt_f64(double a) {
  // CHECK-LABEL: @elementwise_sqrt_f64
  // CHECK: call double @llvm.sqrt.f64(double %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}

double4 elementwise_sqrt_v4f64(double4 a) {
  // CHECK-LABEL: @elementwise_sqrt_v4f64
  // CHECK: call <4 x double> @llvm.sqrt.v4f64(<4 x double> %{{.+}}){{$}}
  return __builtin_elementwise_sqrt(a);
}

#endif

// NODIVOPT: ![[MD_HFDIV]] = !{float 1.000000e+00}
// NODIVOPT: ![[MD_HSQRT]] = !{float 1.500000e+00}
// NODIVOPT: ![[MD_FDIV]] = !{float 2.500000e+00}
// NODIVOPT: ![[MD_SQRT]] = !{float 3.000000e+00}
