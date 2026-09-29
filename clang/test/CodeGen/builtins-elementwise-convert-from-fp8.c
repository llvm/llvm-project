// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -frounding-math -ffp-exception-behavior=strict -DTEST_STRICT_FP -o - %s | FileCheck %s --check-prefixes=CHECK,STRICT

typedef unsigned char uchar4 __attribute__((ext_vector_type(4)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef __bf16 bfloat4 __attribute__((ext_vector_type(4)));
typedef float float4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define {{.*}}@scalar_f8e5m2_f16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 [[BITS]], metadata !"Float8E5M2")
_Float16 scalar_f8e5m2_f16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

// CHECK: declare half @llvm.convert.from.arbitrary.fp.f16.i8(i8, metadata) #[[FP8_ATTRS:[0-9]+]]

// CHECK-LABEL: define {{.*}}@scalar_f8e5m2_bf16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 [[BITS]], metadata !"Float8E5M2")
__bf16 scalar_f8e5m2_bf16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m2_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e5m2_f32(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E5M2")
float scalar_f8e5m2_f32(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e4m3fn_f16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 [[BITS]], metadata !"Float8E4M3FN")
_Float16 scalar_f8e4m3fn_f16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f16(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e4m3fn_bf16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 [[BITS]], metadata !"Float8E4M3FN")
__bf16 scalar_f8e4m3fn_bf16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e4m3fn_f32(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E4M3FN")
float scalar_f8e4m3fn_f32(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e5m3fnu_f16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 [[BITS]], metadata !"Float8E5M3FNU")
_Float16 scalar_f8e5m3fnu_f16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f16(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e5m3fnu_bf16(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 [[BITS]], metadata !"Float8E5M3FNU")
__bf16 scalar_f8e5m3fnu_bf16(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@scalar_f8e5m3fnu_f32(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E5M3FNU")
float scalar_f8e5m3fnu_f32(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m2_f16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M2")
half4 vector_f8e5m2_f16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m2_bf16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x bfloat> @llvm.convert.from.arbitrary.fp.v4bf16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M2")
bfloat4 vector_f8e5m2_bf16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m2_f32(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M2")
float4 vector_f8e5m2_f32(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e4m3fn_f16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E4M3FN")
half4 vector_f8e4m3fn_f16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e4m3fn_bf16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x bfloat> @llvm.convert.from.arbitrary.fp.v4bf16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E4M3FN")
bfloat4 vector_f8e4m3fn_bf16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e4m3fn_f32(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8> [[BITS]], metadata !"Float8E4M3FN")
float4 vector_f8e4m3fn_f32(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m3fnu_f16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M3FNU")
half4 vector_f8e5m3fnu_f16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m3fnu_bf16(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x bfloat> @llvm.convert.from.arbitrary.fp.v4bf16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M3FNU")
bfloat4 vector_f8e5m3fnu_bf16(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@vector_f8e5m3fnu_f32(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M3FNU")
float4 vector_f8e5m3fnu_f32(uchar4 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

// Signedness does not change the interpretation of the bits.
// CHECK-LABEL: define {{.*}}@signed_bits(
// CHECK: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 -68, metadata !"Float8E5M2")
float signed_bits(void) {
  return __builtin_elementwise_convert_from_f8e5m2_f32((signed char)0xbc);
}

// CHECK-LABEL: define {{.*}}@plain_char_bits(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E4M3FN")
float plain_char_bits(char bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(bits);
}

// CHECK-LABEL: define {{.*}}@signed_bitint_bits(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E5M3FNU")
float signed_bitint_bits(_BitInt(8) bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}

// CHECK-LABEL: define {{.*}}@unsigned_bitint_bits(
// CHECK: [[BITS:%.*]] = load i8, ptr
// CHECK-NEXT: call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 [[BITS]], metadata !"Float8E4M3FN")
_Float16 unsigned_bitint_bits(unsigned _BitInt(8) bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f16(bits);
}

typedef _BitInt(8) bitint4 __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define {{.*}}@signed_bitint_vector(
// CHECK: [[BITS:%.*]] = load <4 x i8>, ptr %bits.addr
// CHECK-NEXT: call <4 x bfloat> @llvm.convert.from.arbitrary.fp.v4bf16.v4i8(<4 x i8> [[BITS]], metadata !"Float8E5M3FNU")
bfloat4 signed_bitint_vector(bitint4 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_bf16(bits);
}

typedef signed char schar3 __attribute__((ext_vector_type(3)));
typedef float float3 __attribute__((ext_vector_type(3)));

// CHECK-LABEL: define {{.*}}@three_lanes(
// CHECK: call <3 x float> @llvm.convert.from.arbitrary.fp.v3f32.v3i8(<3 x i8> {{.*}}, metadata !"Float8E4M3FN")
float3 three_lanes(schar3 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_f32(bits);
}

typedef unsigned char uchar1 __attribute__((ext_vector_type(1)));
typedef _Float16 half1 __attribute__((ext_vector_type(1)));

// CHECK-LABEL: define {{.*}}@one_lane(
// CHECK: call <1 x half> @llvm.convert.from.arbitrary.fp.v1f16.v1i8(<1 x i8> {{.*}}, metadata !"Float8E5M2")
half1 one_lane(uchar1 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

typedef signed char gnu_char4 __attribute__((vector_size(4)));
typedef _Float16 gnu_half4 __attribute__((vector_size(8)));
typedef __bf16 gnu_bfloat4 __attribute__((vector_size(8)));
typedef float gnu_float4 __attribute__((vector_size(16)));

// CHECK-LABEL: define {{.*}}@gnu_vector_f16(
// CHECK: call <4 x half> @llvm.convert.from.arbitrary.fp.v4f16.v4i8(<4 x i8> {{.*}}, metadata !"Float8E5M3FNU")
gnu_half4 gnu_vector_f16(gnu_char4 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f16(bits);
}

// CHECK-LABEL: define {{.*}}@gnu_vector_bf16(
// CHECK: call <4 x bfloat> @llvm.convert.from.arbitrary.fp.v4bf16.v4i8(<4 x i8> {{.*}}, metadata !"Float8E4M3FN")
gnu_bfloat4 gnu_vector_bf16(gnu_char4 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@gnu_vector_f32(
// CHECK: call <4 x float> @llvm.convert.from.arbitrary.fp.v4f32.v4i8(<4 x i8> {{.*}}, metadata !"Float8E5M2")
gnu_float4 gnu_vector_f32(gnu_char4 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits);
}

unsigned char next_bits(void);

// CHECK-LABEL: define {{.*}}@evaluate_once(
// CHECK: [[BITS:%.*]] = call {{.*}}i8 @next_bits()
// CHECK-NEXT: [[RESULT:%.*]] = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[BITS]], metadata !"Float8E5M2")
// CHECK-NEXT: ret float [[RESULT]]
float evaluate_once(void) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(next_bits());
}

#ifdef TEST_STRICT_FP
// The conversion ignores the floating-point environment while the addition
// respects the enclosing function's strict floating-point semantics.
// STRICT-LABEL: define {{.*}}@strict_convert(
// STRICT: [[CONVERTED:%.*]] = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 {{.*}}, metadata !"Float8E5M2")
// STRICT-NEXT: call float @llvm.experimental.constrained.fadd.f32(float [[CONVERTED]], float 1.000000e+00, metadata !"round.dynamic", metadata !"fpexcept.strict")
float strict_convert(unsigned char bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f32(bits) + 1.0f;
}
#endif

// CHECK: attributes #[[FP8_ATTRS]] = { {{.*}}nounwind{{.*}}speculatable{{.*}}memory(none){{.*}} }
