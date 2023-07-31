// Test that float variants of sqrt are emitted as available_externally inline
// definitions that call the sqrt intrinsic with appropriate !fpmath metadata
// depending on -cl-fp32-correctly-rounded-divide-sqrt

// Need Matt to help look at this one
// XFAIL: * 

// Test with -fdeclare-opencl-builtins
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -fdeclare-opencl-builtins -finclude-default-header -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -fdeclare-opencl-builtins -finclude-default-header -cl-fp32-correctly-rounded-divide-sqrt -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,CORRECTLYROUNDED %s

// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -fdeclare-opencl-builtins -finclude-default-header -cl-unsafe-math-optimizations -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,DEFAULT-UNSAFE %s
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -fdeclare-opencl-builtins -finclude-default-header -cl-fp32-correctly-rounded-divide-sqrt -cl-unsafe-math-optimizations -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,CORRECTLYROUNDED-UNSAFE %s

// Test without -fdeclare-opencl-builtins
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -finclude-default-header -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,DEFAULT %s
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -finclude-default-header -cl-fp32-correctly-rounded-divide-sqrt -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,CORRECTLYROUNDED %s

// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -finclude-default-header -cl-unsafe-math-optimizations -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,DEFAULT-UNSAFE %s
// RUN: %clang_cc1 -disable-llvm-passes -triple amdgcn-unknown-unknown -finclude-default-header -cl-fp32-correctly-rounded-divide-sqrt -cl-unsafe-math-optimizations -S -emit-llvm -o - %s | FileCheck -check-prefixes=CHECK,CORRECTLYROUNDED-UNSAFE %s

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// CxHECK-LABEL: define {{.*}} float @call_sqrt_f32(
// CxHECK: call {{.*}} float @_Z4sqrtf(float noundef %{{.+}}) #{{[0-9]+$}}
float call_sqrt_f32(float x) {
  return sqrt(x);
}

// CHECK-LABEL: define dso_local float @call_sqrt_f32(float noundef %x)
// DEFAULT: call float @llvm.sqrt.f32(float %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call float @llvm.sqrt.f32(float %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn float @llvm.sqrt.f32(float %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn float @llvm.sqrt.f32(float %{{.+}}){{$}}

// CHECK-LABEL: define {{.*}} <2 x float> @call_sqrt_v2f32(
// CHECK: call {{.*}} <2 x float> @_Z4sqrtDv2_f(<2 x float> noundef %{{.*}}) #{{[0-9]+$}}
float2 call_sqrt_v2f32(float2 x) {
  return sqrt(x);
}

// CHECK-LABEL: define available_externally <2 x float> @_Z4sqrtDv2_f(<2 x float> noundef %__x)
// DEFAULT: call <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn <2 x float> @llvm.sqrt.v2f32(<2 x float> %{{.+}}){{$}}

// CHECK-LABEL: define {{.*}} <3 x float> @call_sqrt_v3f32(
// CHECK: call {{.*}} <3 x float> @_Z4sqrtDv3_f(<3 x float> noundef %{{.*}}) #{{[0-9]+$}}
float3 call_sqrt_v3f32(float3 x) {
  return sqrt(x);
}

// CHECK-LABEL: define available_externally <3 x float> @_Z4sqrtDv3_f(<3 x float> noundef %__x)
// DEFAULT: call <3 x float> @llvm.sqrt.v3f32(<3 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call <3 x float> @llvm.sqrt.v3f32(<3 x float> %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn <3 x float> @llvm.sqrt.v3f32(<3 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn <3 x float> @llvm.sqrt.v3f32(<3 x float> %{{.+}}){{$}}


// CHECK-LABEL: define {{.*}} <4 x float> @call_sqrt_v4f32(
// CHECK: call {{.*}} <4 x float> @_Z4sqrtDv4_f(<4 x float> noundef %{{.*}}) #{{[0-9]+$}}
float4 call_sqrt_v4f32(float4 x) {
  return sqrt(x);
}

// CHECK-LABEL: define available_externally <4 x float> @_Z4sqrtDv4_f(<4 x float> noundef %__x)
// DEFAULT: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn <4 x float> @llvm.sqrt.v4f32(<4 x float> %{{.+}}){{$}}

// CHECK-LABEL: define {{.*}} <8 x float> @call_sqrt_v8f32(
// CHECK: call {{.*}} <8 x float> @_Z4sqrtDv8_f(<8 x float> noundef %{{.*}}) #{{[0-9]+$}}
float8 call_sqrt_v8f32(float8 x) {
  return sqrt(x);
}

// CHECK-LABEL: define available_externally <8 x float> @_Z4sqrtDv8_f(<8 x float> noundef %__x)
// DEFAULT: call <8 x float> @llvm.sqrt.v8f32(<8 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call <8 x float> @llvm.sqrt.v8f32(<8 x float> %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn <8 x float> @llvm.sqrt.v8f32(<8 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn <8 x float> @llvm.sqrt.v8f32(<8 x float> %{{.+}}){{$}}


// CHECK-LABEL: define {{.*}} <16 x float> @call_sqrt_v16f32(
// CHECK: call {{.*}} <16 x float> @_Z4sqrtDv16_f(<16 x float> noundef %{{.*}}) #{{[0-9]+$}}
float16 call_sqrt_v16f32(float16 x) {
  return sqrt(x);
}

// CHECK-LABEL: define available_externally <16 x float> @_Z4sqrtDv16_f(<16 x float> noundef %__x)
// DEFAULT: call <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED: call <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.+}}){{$}}

// DEFAULT-UNSAFE: call reassoc nsz arcp contract afn <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.+}}), !fpmath [[$FPMATH:![0-9]+]]{{$}}
// CORRECTLYROUNDED-UNSAFE: call reassoc nsz arcp contract afn <16 x float> @llvm.sqrt.v16f32(<16 x float> %{{.+}}){{$}}


// Not for f64
// CHECK-LABEL: define {{.*}} double @call_sqrt_f64(
// CHECK: call {{.*}} double @_Z4sqrtd(double noundef %{{.+}}) #{{[0-9]+$}}
double call_sqrt_f64(double x) {
  return sqrt(x);
}

// CHECK-NOT: define

// Not for f64
// CHECK-LABEL: define {{.*}} <2 x double> @call_sqrt_v2f64(
// CHECK: call {{.*}} <2 x double> @_Z4sqrtDv2_d(<2 x double> noundef %{{.+}}) #{{[0-9]+$}}
double2 call_sqrt_v2f64(double2 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <3 x double> @call_sqrt_v3f64(
// CHECK: call {{.*}} <3 x double> @_Z4sqrtDv3_d(<3 x double> noundef %{{.+}}) #{{[0-9]+$}}
double3 call_sqrt_v3f64(double3 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <4 x double> @call_sqrt_v4f64(
// CHECK: call {{.*}} <4 x double> @_Z4sqrtDv4_d(<4 x double> noundef %{{.+}}) #{{[0-9]+$}}
double4 call_sqrt_v4f64(double4 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <8 x double> @call_sqrt_v8f64(
// CHECK: call {{.*}} <8 x double> @_Z4sqrtDv8_d(<8 x double> noundef %{{.+}}) #{{[0-9]+$}}
double8 call_sqrt_v8f64(double8 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <16 x double> @call_sqrt_v16f64(
// CHECK: call {{.*}} <16 x double> @_Z4sqrtDv16_d(<16 x double> noundef %{{.+}}) #{{[0-9]+$}}
double16 call_sqrt_v16f64(double16 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// Not for f16
// CHECK-LABEL: define {{.*}} half @call_sqrt_f16(
// CHECK: call {{.*}} half @_Z4sqrtDh(half noundef %{{.+}}) #{{[0-9]+$}}
half call_sqrt_f16(half x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <2 x half> @call_sqrt_v2f16(
// CHECK: call {{.*}} <2 x half> @_Z4sqrtDv2_Dh(<2 x half> noundef %{{.+}}) #{{[0-9]+$}}
half2 call_sqrt_v2f16(half2 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <3 x half> @call_sqrt_v3f16(
// CHECK: call {{.*}} <3 x half> @_Z4sqrtDv3_Dh(<3 x half> noundef %{{.+}}) #{{[0-9]+$}}
half3 call_sqrt_v3f16(half3 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <4 x half> @call_sqrt_v4f16(
// CHECK: call {{.*}} <4 x half> @_Z4sqrtDv4_Dh(<4 x half> noundef %{{.+}}) #{{[0-9]+$}}
half4 call_sqrt_v4f16(half4 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <8 x half> @call_sqrt_v8f16(
// CHECK: call {{.*}} <8 x half> @_Z4sqrtDv8_Dh(<8 x half> noundef %{{.+}}) #{{[0-9]+$}}
half8 call_sqrt_v8f16(half8 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// CHECK-LABEL: define {{.*}} <16 x half> @call_sqrt_v16f16(
// CHECK: call {{.*}} <16 x half> @_Z4sqrtDv16_Dh(<16 x half> noundef %{{.+}}) #{{[0-9]+$}}
half16 call_sqrt_v16f16(half16 x) {
  return sqrt(x);
}

// CHECK-NOT: define

// DEFAULT: [[$FPMATH]] = !{float 3.000000e+00}
