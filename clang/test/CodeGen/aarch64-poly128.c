// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN: -disable-O0-optnone -ffp-contract=fast -emit-llvm -o - %s | opt -S -mem2reg \
// RUN:  | FileCheck %s

// Test new aarch64 intrinsics with poly128
// FIXME: Currently, poly128_t equals to uint128, which will be spilt into
// two 64-bit GPR(eg X0, X1). Now moving data from X0, X1 to FPR128 will
// introduce 2 store and 1 load instructions(store X0, X1 to memory and
// then load back to Q0). If target has NEON, this is better replaced by
// FMOV or INS.

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} void @test_vstrq_p128(i128* %ptr, i128 %val) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i128* %ptr to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i128*
// CHECK:   store i128 %val, i128* [[TMP1]]
// CHECK:   ret void
void test_vstrq_p128(poly128_t * ptr, poly128_t val) {
  vstrq_p128(ptr, val);

}

// CHECK-LABEL: define{{.*}} i128 @test_vldrq_p128(i128* %ptr) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i128* %ptr to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i128*
// CHECK:   [[TMP2:%.*]] = load i128, i128* [[TMP1]]
// CHECK:   ret i128 [[TMP2]]
poly128_t test_vldrq_p128(poly128_t * ptr) {
  return vldrq_p128(ptr);

}

// CHECK-LABEL: define{{.*}} void @test_ld_st_p128(i128* %ptr) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i128* %ptr to i8*
// CHECK:   [[TMP1:%.*]] = bitcast i8* [[TMP0]] to i128*
// CHECK:   [[TMP2:%.*]] = load i128, i128* [[TMP1]]
// CHECK:   [[ADD_PTR:%.*]] = getelementptr inbounds i128, i128* %ptr, i64 1
// CHECK:   [[TMP3:%.*]] = bitcast i128* [[ADD_PTR]] to i8*
// CHECK:   [[TMP4:%.*]] = bitcast i8* [[TMP3]] to i128*
// CHECK:   store i128 [[TMP2]], i128* [[TMP4]]
// CHECK:   ret void
void test_ld_st_p128(poly128_t * ptr) {
   vstrq_p128(ptr+1, vldrq_p128(ptr));

}

// CHECK-LABEL: define{{.*}} i128 @test_vmull_p64(i64 %a, i64 %b) #0 {
// CHECK:   [[VMULL_P64_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 %a, i64 %b) #3
// CHECK:   [[VMULL_P641_I:%.*]] = bitcast <16 x i8> [[VMULL_P64_I]] to i128
// CHECK:   ret i128 [[VMULL_P641_I]]
poly128_t test_vmull_p64(poly64_t a, poly64_t b) {
  return vmull_p64(a, b);
}

// CHECK-LABEL: define{{.*}} i128 @test_vmull_high_p64(<2 x i64> %a, <2 x i64> %b) #1 {
// CHECK:   [[SHUFFLE_I_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %a, <1 x i32> <i32 1>
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> [[SHUFFLE_I_I]] to i64
// CHECK:   [[SHUFFLE_I7_I:%.*]] = shufflevector <2 x i64> %b, <2 x i64> %b, <1 x i32> <i32 1>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> [[SHUFFLE_I7_I]] to i64
// CHECK:   [[VMULL_P64_I_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.pmull64(i64 [[TMP0]], i64 [[TMP1]]) #3
// CHECK:   [[VMULL_P641_I_I:%.*]] = bitcast <16 x i8> [[VMULL_P64_I_I]] to i128
// CHECK:   ret i128 [[VMULL_P641_I_I]]
poly128_t test_vmull_high_p64(poly64x2_t a, poly64x2_t b) {
  return vmull_high_p64(a, b);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_s8(<16 x i8> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_s8(int8x16_t a) {
  return vreinterpretq_p128_s8(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_s16(<8 x i16> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_s16(int16x8_t a) {
  return vreinterpretq_p128_s16(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_s32(<4 x i32> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_s32(int32x4_t a) {
  return vreinterpretq_p128_s32(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_s64(<2 x i64> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_s64(int64x2_t a) {
  return vreinterpretq_p128_s64(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_u8(<16 x i8> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_u8(uint8x16_t a) {
  return vreinterpretq_p128_u8(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_u16(<8 x i16> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_u16(uint16x8_t a) {
  return vreinterpretq_p128_u16(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_u32(<4 x i32> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x i32> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_u32(uint32x4_t a) {
  return vreinterpretq_p128_u32(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_u64(<2 x i64> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_u64(uint64x2_t a) {
  return vreinterpretq_p128_u64(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_f32(<4 x float> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <4 x float> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_f32(float32x4_t a) {
  return vreinterpretq_p128_f32(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_f64(<2 x double> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x double> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_f64(float64x2_t a) {
  return vreinterpretq_p128_f64(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_p8(<16 x i8> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <16 x i8> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_p8(poly8x16_t a) {
  return vreinterpretq_p128_p8(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_p16(<8 x i16> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <8 x i16> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_p16(poly16x8_t a) {
  return vreinterpretq_p128_p16(a);
}

// CHECK-LABEL: define{{.*}} i128 @test_vreinterpretq_p128_p64(<2 x i64> %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to i128
// CHECK:   ret i128 [[TMP0]]
poly128_t test_vreinterpretq_p128_p64(poly64x2_t a) {
  return vreinterpretq_p128_p64(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vreinterpretq_s8_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
int8x16_t test_vreinterpretq_s8_p128(poly128_t a) {
  return vreinterpretq_s8_p128(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vreinterpretq_s16_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
int16x8_t test_vreinterpretq_s16_p128(poly128_t  a) {
  return vreinterpretq_s16_p128(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vreinterpretq_s32_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
int32x4_t test_vreinterpretq_s32_p128(poly128_t a) {
  return vreinterpretq_s32_p128(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vreinterpretq_s64_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
int64x2_t test_vreinterpretq_s64_p128(poly128_t  a) {
  return vreinterpretq_s64_p128(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vreinterpretq_u8_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
uint8x16_t test_vreinterpretq_u8_p128(poly128_t  a) {
  return vreinterpretq_u8_p128(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vreinterpretq_u16_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
uint16x8_t test_vreinterpretq_u16_p128(poly128_t  a) {
  return vreinterpretq_u16_p128(a);
}

// CHECK-LABEL: define{{.*}} <4 x i32> @test_vreinterpretq_u32_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <4 x i32>
// CHECK:   ret <4 x i32> [[TMP0]]
uint32x4_t test_vreinterpretq_u32_p128(poly128_t  a) {
  return vreinterpretq_u32_p128(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vreinterpretq_u64_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
uint64x2_t test_vreinterpretq_u64_p128(poly128_t  a) {
  return vreinterpretq_u64_p128(a);
}

// CHECK-LABEL: define{{.*}} <4 x float> @test_vreinterpretq_f32_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <4 x float>
// CHECK:   ret <4 x float> [[TMP0]]
float32x4_t test_vreinterpretq_f32_p128(poly128_t  a) {
  return vreinterpretq_f32_p128(a);
}

// CHECK-LABEL: define{{.*}} <2 x double> @test_vreinterpretq_f64_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <2 x double>
// CHECK:   ret <2 x double> [[TMP0]]
float64x2_t test_vreinterpretq_f64_p128(poly128_t  a) {
  return vreinterpretq_f64_p128(a);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vreinterpretq_p8_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <16 x i8>
// CHECK:   ret <16 x i8> [[TMP0]]
poly8x16_t test_vreinterpretq_p8_p128(poly128_t  a) {
  return vreinterpretq_p8_p128(a);
}

// CHECK-LABEL: define{{.*}} <8 x i16> @test_vreinterpretq_p16_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <8 x i16>
// CHECK:   ret <8 x i16> [[TMP0]]
poly16x8_t test_vreinterpretq_p16_p128(poly128_t  a) {
  return vreinterpretq_p16_p128(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vreinterpretq_p64_p128(i128 %a) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast i128 %a to <2 x i64>
// CHECK:   ret <2 x i64> [[TMP0]]
poly64x2_t test_vreinterpretq_p64_p128(poly128_t  a) {
  return vreinterpretq_p64_p128(a);
}


