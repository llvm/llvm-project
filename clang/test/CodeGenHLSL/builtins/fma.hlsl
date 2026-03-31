// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s

// CHECK-LABEL: define {{.*}} double @{{.*}}fma_double{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn double @llvm.fma.f64(double
// CHECK: ret double
double fma_double(double a, double b, double c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <2 x double> @{{.*}}fma_double2{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.fma.v2f64(<2 x double>
// CHECK: ret <2 x double>
double2 fma_double2(double2 a, double2 b, double2 c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <3 x double> @{{.*}}fma_double3{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.fma.v3f64(<3 x double>
// CHECK: ret <3 x double>
double3 fma_double3(double3 a, double3 b, double3 c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}fma_double4{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.fma.v4f64(<4 x double>
// CHECK: ret <4 x double>
double4 fma_double4(double4 a, double4 b, double4 c) { return fma(a, b, c); }

// CHECK-LABEL: define {{.*}} <1 x double> @{{.*}}fma_double1x1{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <1 x double> @llvm.fma.v1f64(<1 x double>
// CHECK: ret <1 x double>
double1x1 fma_double1x1(double1x1 a, double1x1 b, double1x1 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <2 x double> @{{.*}}fma_double1x2{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.fma.v2f64(<2 x double>
// CHECK: ret <2 x double>
double1x2 fma_double1x2(double1x2 a, double1x2 b, double1x2 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <3 x double> @{{.*}}fma_double1x3{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.fma.v3f64(<3 x double>
// CHECK: ret <3 x double>
double1x3 fma_double1x3(double1x3 a, double1x3 b, double1x3 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}fma_double1x4{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.fma.v4f64(<4 x double>
// CHECK: ret <4 x double>
double1x4 fma_double1x4(double1x4 a, double1x4 b, double1x4 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <2 x double> @{{.*}}fma_double2x1{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <2 x double> @llvm.fma.v2f64(<2 x double>
// CHECK: ret <2 x double>
double2x1 fma_double2x1(double2x1 a, double2x1 b, double2x1 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}fma_double2x2{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.fma.v4f64(<4 x double>
// CHECK: ret <4 x double>
double2x2 fma_double2x2(double2x2 a, double2x2 b, double2x2 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <6 x double> @{{.*}}fma_double2x3{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.fma.v6f64(<6 x double>
// CHECK: ret <6 x double>
double2x3 fma_double2x3(double2x3 a, double2x3 b, double2x3 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <8 x double> @{{.*}}fma_double2x4{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x double> @llvm.fma.v8f64(<8 x double>
// CHECK: ret <8 x double>
double2x4 fma_double2x4(double2x4 a, double2x4 b, double2x4 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <3 x double> @{{.*}}fma_double3x1{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.fma.v3f64(<3 x double>
// CHECK: ret <3 x double>
double3x1 fma_double3x1(double3x1 a, double3x1 b, double3x1 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <6 x double> @{{.*}}fma_double3x2{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <6 x double> @llvm.fma.v6f64(<6 x double>
// CHECK: ret <6 x double>
double3x2 fma_double3x2(double3x2 a, double3x2 b, double3x2 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <9 x double> @{{.*}}fma_double3x3{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <9 x double> @llvm.fma.v9f64(<9 x double>
// CHECK: ret <9 x double>
double3x3 fma_double3x3(double3x3 a, double3x3 b, double3x3 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <12 x double> @{{.*}}fma_double3x4{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x double> @llvm.fma.v12f64(<12 x double>
// CHECK: ret <12 x double>
double3x4 fma_double3x4(double3x4 a, double3x4 b, double3x4 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <4 x double> @{{.*}}fma_double4x1{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <4 x double> @llvm.fma.v4f64(<4 x double>
// CHECK: ret <4 x double>
double4x1 fma_double4x1(double4x1 a, double4x1 b, double4x1 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <8 x double> @{{.*}}fma_double4x2{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <8 x double> @llvm.fma.v8f64(<8 x double>
// CHECK: ret <8 x double>
double4x2 fma_double4x2(double4x2 a, double4x2 b, double4x2 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <12 x double> @{{.*}}fma_double4x3{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <12 x double> @llvm.fma.v12f64(<12 x double>
// CHECK: ret <12 x double>
double4x3 fma_double4x3(double4x3 a, double4x3 b, double4x3 c) {
  return fma(a, b, c);
}

// CHECK-LABEL: define {{.*}} <16 x double> @{{.*}}fma_double4x4{{.*}}(
// CHECK: call reassoc nnan ninf nsz arcp afn <16 x double> @llvm.fma.v16f64(<16 x double>
// CHECK: ret <16 x double>
double4x4 fma_double4x4(double4x4 a, double4x4 b, double4x4 c) {
  return fma(a, b, c);
}
