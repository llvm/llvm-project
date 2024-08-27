// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - -DSPIRV | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -finclude-default-header -fnative-half-type -emit-llvm -disable-llvm-passes  %s -o - | FileCheck %s

// Test explicit matrix casts.
// This is adapted to HLSL from CodeGen/matrix-cast.c.

// CHECK-LABEL: define {{.*}}cast_int16_matrix_to_int
void cast_int16_matrix_to_int() {
  int16_t4x4 c;
  int4x4 i;

  // CHECK:       [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (int4x4)c;
}

// CHECK-LABEL: define {{.*}}cast_int16_matrix_to_uint
void cast_int16_matrix_to_uint() {
  int16_t4x4 c;
  uint4x4 u;
  // CHECK:       [[C:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i16> [[C]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  u = (uint4x4)c;
}

// CHECK-LABEL: define {{.*}}cast_uint64_matrix_to_int16
void cast_uint64_matrix_to_int16() {
  uint64_t4x4 u;
  int16_t4x4 s;
  // CHECK:       [[U:%.*]] = load <16 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i64> [[U]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (int16_t4x4)u;
}

// CHECK-LABEL: define {{.*}}cast_int_matrix_to_int16
void cast_int_matrix_to_int16() {
  int4x4 i;
  int16_t4x4 s;
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i32> [[I]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (int16_t4x4)i;
}

// CHECK-LABEL: define {{.*}}cast_int_matrix_to_float
void cast_int_matrix_to_float() {
  int4x4 i;
  float4x4 f;
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sitofp <16 x i32> [[I]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (float4x4)i;
}

// CHECK-LABEL: define {{.*}}cast_uint_matrix_to_float
void cast_uint_matrix_to_float() {
  uint16_t4x4 u;
  float4x4 f;
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = uitofp <16 x i16> [[U]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (float4x4)u;
}

// CHECK-LABEL: define {{.*}}cast_double_matrix_to_int
void cast_double_matrix_to_int() {
  double4x4 d;
  int4x4 i;
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptosi <16 x double> [[D]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (int4x4)d;
}

// CHECK-LABEL: define {{.*}}cast_float_matrix_to_uint16
void cast_float_matrix_to_uint16() {
  float4x4 f;
  uint16_t4x4 i;
  // CHECK:       [[F:%.*]] = load <16 x float>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = fptoui <16 x float> [[F]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  i = (uint16_t4x4)f;
}

// CHECK-LABEL: define {{.*}}cast_double_matrix_to_float
void cast_double_matrix_to_float() {
  double4x4 d;
  float4x4 f;
  // CHECK:       [[D:%.*]] = load <16 x double>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = fptrunc <16 x double> [[D]] to <16 x float>
  // CHECK-NEXT:  store <16 x float> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  f = (float4x4)d;
}

// CHECK-LABEL: define {{.*}}cast_uint16_to_uint
void cast_uint16_to_uint() {
  uint16_t4x4 s;
  uint4x4 i;
  // CHECK:       [[S:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[S]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (uint4x4)s;
}

// CHECK-LABEL: define {{.*}}cast_uint64_to_uint16
void cast_uint64_to_uint16() {
  uint64_t4x4 l;
  uint16_t4x4 s;
  // CHECK:       [[L:%.*]] = load <16 x i64>, ptr {{.*}}, align 8
  // CHECK-NEXT:  [[CONV:%.*]] = trunc <16 x i64> [[L]] to <16 x i16>
  // CHECK-NEXT:  store <16 x i16> [[CONV]], ptr {{.*}}, align 2
  // CHECK-NEXT:  ret void

  s = (uint16_t4x4)l;
}

// CHECK-LABEL: define {{.*}}cast_uint16_to_int
void cast_uint16_to_int() {
  uint16_t4x4 u;
  int4x4 i;
  // CHECK:       [[U:%.*]] = load <16 x i16>, ptr {{.*}}, align 2
  // CHECK-NEXT:  [[CONV:%.*]] = zext <16 x i16> [[U]] to <16 x i32>
  // CHECK-NEXT:  store <16 x i32> [[CONV]], ptr {{.*}}, align 4
  // CHECK-NEXT:  ret void

  i = (int4x4)u;
}

// CHECK-LABEL: define {{.*}}cast_int_to_uint64
void cast_int_to_uint64() {
  int4x4 i;
  uint64_t4x4 u;
  // CHECK:       [[I:%.*]] = load <16 x i32>, ptr {{.*}}, align 4
  // CHECK-NEXT:  [[CONV:%.*]] = sext <16 x i32> [[I]] to <16 x i64>
  // CHECK-NEXT:  store <16 x i64> [[CONV]], ptr {{.*}}, align 8
  // CHECK-NEXT:  ret void

  u = (uint64_t4x4)i;
}
