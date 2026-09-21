// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// REQUIRES: aarch64-registered-target

// __mfp8 is a scalar source even though its IR representation is <1 x i8>.
// CHECK-LABEL: define {{.*}}@mfp8_f16(
// CHECK: [[BITS:%.*]] = load <1 x i8>, ptr
// CHECK-NEXT: [[SCALAR:%.*]] = bitcast <1 x i8> [[BITS]] to i8
// CHECK-NEXT: call half @llvm.convert.from.arbitrary.fp.f16.i8(i8 [[SCALAR]], metadata !"Float8E5M2")
_Float16 mfp8_f16(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e5m2_f16(bits);
}

// CHECK-LABEL: define {{.*}}@mfp8_bf16(
// CHECK: [[BITS:%.*]] = load <1 x i8>, ptr
// CHECK-NEXT: [[SCALAR:%.*]] = bitcast <1 x i8> [[BITS]] to i8
// CHECK-NEXT: call bfloat @llvm.convert.from.arbitrary.fp.bf16.i8(i8 [[SCALAR]], metadata !"Float8E4M3FN")
__bf16 mfp8_bf16(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits);
}

// CHECK-LABEL: define {{.*}}@mfp8_f32(
// CHECK: [[BITS:%.*]] = load <1 x i8>, ptr
// CHECK-NEXT: [[SCALAR:%.*]] = bitcast <1 x i8> [[BITS]] to i8
// CHECK-NEXT: call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 [[SCALAR]], metadata !"Float8E5M3FNU")
float mfp8_f32(__mfp8 bits) {
  return __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits);
}
