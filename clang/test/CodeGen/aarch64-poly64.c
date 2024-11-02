// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN:  -ffp-contract=fast -disable-O0-optnone -emit-llvm -o - %s | opt -S -passes=mem2reg \
// RUN:  | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vceq_p64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[CMP_I:%.*]] = icmp eq <1 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <1 x i1> [[CMP_I]] to <1 x i64>
// CHECK:   ret <1 x i64> [[SEXT_I]]
uint64x1_t test_vceq_p64(poly64x1_t a, poly64x1_t b) {
  return vceq_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vceqq_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[CMP_I:%.*]] = icmp eq <2 x i64> %a, %b
// CHECK:   [[SEXT_I:%.*]] = sext <2 x i1> [[CMP_I]] to <2 x i64>
// CHECK:   ret <2 x i64> [[SEXT_I]]
uint64x2_t test_vceqq_p64(poly64x2_t a, poly64x2_t b) {
  return vceqq_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vtst_p64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[TMP4:%.*]] = and <1 x i64> %a, %b
// CHECK:   [[TMP5:%.*]] = icmp ne <1 x i64> [[TMP4]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <1 x i1> [[TMP5]] to <1 x i64>
// CHECK:   ret <1 x i64> [[VTST_I]]
uint64x1_t test_vtst_p64(poly64x1_t a, poly64x1_t b) {
  return vtst_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vtstq_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[TMP4:%.*]] = and <2 x i64> %a, %b
// CHECK:   [[TMP5:%.*]] = icmp ne <2 x i64> [[TMP4]], zeroinitializer
// CHECK:   [[VTST_I:%.*]] = sext <2 x i1> [[TMP5]] to <2 x i64>
// CHECK:   ret <2 x i64> [[VTST_I]]
uint64x2_t test_vtstq_p64(poly64x2_t a, poly64x2_t b) {
  return vtstq_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vbsl_p64(<1 x i64> noundef %a, <1 x i64> noundef %b, <1 x i64> noundef %c) #0 {
// CHECK:   [[VBSL3_I:%.*]] = and <1 x i64> %a, %b
// CHECK:   [[TMP3:%.*]] = xor <1 x i64> %a, <i64 -1>
// CHECK:   [[VBSL4_I:%.*]] = and <1 x i64> [[TMP3]], %c
// CHECK:   [[VBSL5_I:%.*]] = or <1 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <1 x i64> [[VBSL5_I]]
poly64x1_t test_vbsl_p64(poly64x1_t a, poly64x1_t b, poly64x1_t c) {
  return vbsl_p64(a, b, c);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vbslq_p64(<2 x i64> noundef %a, <2 x i64> noundef %b, <2 x i64> noundef %c) #1 {
// CHECK:   [[VBSL3_I:%.*]] = and <2 x i64> %a, %b
// CHECK:   [[TMP3:%.*]] = xor <2 x i64> %a, <i64 -1, i64 -1>
// CHECK:   [[VBSL4_I:%.*]] = and <2 x i64> [[TMP3]], %c
// CHECK:   [[VBSL5_I:%.*]] = or <2 x i64> [[VBSL3_I]], [[VBSL4_I]]
// CHECK:   ret <2 x i64> [[VBSL5_I]]
poly64x2_t test_vbslq_p64(poly64x2_t a, poly64x2_t b, poly64x2_t c) {
  return vbslq_p64(a, b, c);
}

// CHECK-LABEL: define{{.*}} i64 @test_vget_lane_p64(<1 x i64> noundef %v) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %v, i32 0
// CHECK:   ret i64 [[VGET_LANE]]
poly64_t test_vget_lane_p64(poly64x1_t v) {
  return vget_lane_p64(v, 0);
}

// CHECK-LABEL: define{{.*}} i64 @test_vgetq_lane_p64(<2 x i64> noundef %v) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %v, i32 1
// CHECK:   ret i64 [[VGETQ_LANE]]
poly64_t test_vgetq_lane_p64(poly64x2_t v) {
  return vgetq_lane_p64(v, 1);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vset_lane_p64(i64 noundef %a, <1 x i64> noundef %v) #0 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x i64> %v, i64 %a, i32 0
// CHECK:   ret <1 x i64> [[VSET_LANE]]
poly64x1_t test_vset_lane_p64(poly64_t a, poly64x1_t v) {
  return vset_lane_p64(a, v, 0);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vsetq_lane_p64(i64 noundef %a, <2 x i64> noundef %v) #1 {
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %v, i64 %a, i32 1
// CHECK:   ret <2 x i64> [[VSET_LANE]]
poly64x2_t test_vsetq_lane_p64(poly64_t a, poly64x2_t v) {
  return vsetq_lane_p64(a, v, 1);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vcopy_lane_p64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %b, i32 0
// CHECK:   [[VSET_LANE:%.*]] = insertelement <1 x i64> %a, i64 [[VGET_LANE]], i32 0
// CHECK:   ret <1 x i64> [[VSET_LANE]]
poly64x1_t test_vcopy_lane_p64(poly64x1_t a, poly64x1_t b) {
  return vcopy_lane_p64(a, 0, b, 0);

}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcopyq_lane_p64(<2 x i64> noundef %a, <1 x i64> noundef %b) #1 {
// CHECK:   [[VGET_LANE:%.*]] = extractelement <1 x i64> %b, i32 0
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %a, i64 [[VGET_LANE]], i32 1
// CHECK:   ret <2 x i64> [[VSET_LANE]]
poly64x2_t test_vcopyq_lane_p64(poly64x2_t a, poly64x1_t b) {
  return vcopyq_lane_p64(a, 1, b, 0);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcopyq_laneq_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[VGETQ_LANE:%.*]] = extractelement <2 x i64> %b, i32 1
// CHECK:   [[VSET_LANE:%.*]] = insertelement <2 x i64> %a, i64 [[VGETQ_LANE]], i32 1
// CHECK:   ret <2 x i64> [[VSET_LANE]]
poly64x2_t test_vcopyq_laneq_p64(poly64x2_t a, poly64x2_t b) {
  return vcopyq_laneq_p64(a, 1, b, 1);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vcreate_p64(i64 noundef %a) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast i64 %a to <1 x i64>
// CHECK:   ret <1 x i64> [[TMP0]]
poly64x1_t test_vcreate_p64(uint64_t a) {
  return vcreate_p64(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vdup_n_p64(i64 noundef %a) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <1 x i64> undef, i64 %a, i32 0
// CHECK:   ret <1 x i64> [[VECINIT_I]]
poly64x1_t test_vdup_n_p64(poly64_t a) {
  return vdup_n_p64(a);
}
// CHECK-LABEL: define{{.*}} <2 x i64> @test_vdupq_n_p64(i64 noundef %a) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i64> undef, i64 %a, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i64> [[VECINIT_I]], i64 %a, i32 1
// CHECK:   ret <2 x i64> [[VECINIT1_I]]
poly64x2_t test_vdupq_n_p64(poly64_t a) {
  return vdupq_n_p64(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vmov_n_p64(i64 noundef %a) #0 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <1 x i64> undef, i64 %a, i32 0
// CHECK:   ret <1 x i64> [[VECINIT_I]]
poly64x1_t test_vmov_n_p64(poly64_t a) {
  return vmov_n_p64(a);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vmovq_n_p64(i64 noundef %a) #1 {
// CHECK:   [[VECINIT_I:%.*]] = insertelement <2 x i64> undef, i64 %a, i32 0
// CHECK:   [[VECINIT1_I:%.*]] = insertelement <2 x i64> [[VECINIT_I]], i64 %a, i32 1
// CHECK:   ret <2 x i64> [[VECINIT1_I]]
poly64x2_t test_vmovq_n_p64(poly64_t a) {
  return vmovq_n_p64(a);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vdup_lane_p64(<1 x i64> noundef %vec) #0 {
// CHECK:    [[TMP0:%.*]] = bitcast <1 x i64> [[VEC:%.*]] to <8 x i8>
// CHECK:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:    [[LANE:%.*]] = shufflevector <1 x i64> [[TMP1]], <1 x i64> [[TMP1]], <1 x i32> zeroinitializer
// CHECK:    ret <1 x i64> [[LANE]]
poly64x1_t test_vdup_lane_p64(poly64x1_t vec) {
  return vdup_lane_p64(vec, 0);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vdupq_lane_p64(<1 x i64> noundef %vec) #1 {
// CHECK:    [[TMP0:%.*]] = bitcast <1 x i64> [[VEC:%.*]] to <8 x i8>
// CHECK:    [[TMP1:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:    [[LANE:%.*]] = shufflevector <1 x i64> [[TMP1]], <1 x i64> [[TMP1]], <2 x i32> zeroinitializer
// CHECK:    ret <2 x i64> [[LANE]]
poly64x2_t test_vdupq_lane_p64(poly64x1_t vec) {
  return vdupq_lane_p64(vec, 0);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vdupq_laneq_p64(<2 x i64> noundef %vec) #1 {
// CHECK:    [[TMP0:%.*]] = bitcast <2 x i64> [[VEC:%.*]] to <16 x i8>
// CHECK:    [[TMP1:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:    [[LANE:%.*]] = shufflevector <2 x i64> [[TMP1]], <2 x i64> [[TMP1]], <2 x i32> <i32 1, i32 1>
// CHECK:    ret <2 x i64> [[LANE]]
poly64x2_t test_vdupq_laneq_p64(poly64x2_t vec) {
  return vdupq_laneq_p64(vec, 1);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vcombine_p64(<1 x i64> noundef %low, <1 x i64> noundef %high) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <1 x i64> %low, <1 x i64> %high, <2 x i32> <i32 0, i32 1>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vcombine_p64(poly64x1_t low, poly64x1_t high) {
  return vcombine_p64(low, high);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vld1_p64(ptr noundef %ptr) #0 {
// CHECK:   [[TMP2:%.*]] = load <1 x i64>, ptr %ptr
// CHECK:   ret <1 x i64> [[TMP2]]
poly64x1_t test_vld1_p64(poly64_t const * ptr) {
  return vld1_p64(ptr);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vld1q_p64(ptr noundef %ptr) #1 {
// CHECK:   [[TMP2:%.*]] = load <2 x i64>, ptr %ptr
// CHECK:   ret <2 x i64> [[TMP2]]
poly64x2_t test_vld1q_p64(poly64_t const * ptr) {
  return vld1q_p64(ptr);
}

// CHECK-LABEL: define{{.*}} void @test_vst1_p64(ptr noundef %ptr, <1 x i64> noundef %val) #0 {
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %val to <8 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   store <1 x i64> [[TMP3]], ptr %ptr
// CHECK:   ret void
void test_vst1_p64(poly64_t * ptr, poly64x1_t val) {
  return vst1_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst1q_p64(ptr noundef %ptr, <2 x i64> noundef %val) #1 {
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %val to <16 x i8>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   store <2 x i64> [[TMP3]], ptr %ptr
// CHECK:   ret void
void test_vst1q_p64(poly64_t * ptr, poly64x2_t val) {
  return vst1q_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x2_t @test_vld2_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[VLD2:%.*]] = call { <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld2.v1i64.p0(ptr %ptr)
// CHECK:   store { <1 x i64>, <1 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 16, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x2_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x2_t [[TMP6]]
poly64x1x2_t test_vld2_p64(poly64_t const * ptr) {
  return vld2_p64(ptr);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x2_t @test_vld2q_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[VLD2:%.*]] = call { <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld2.v2i64.p0(ptr %ptr)
// CHECK:   store { <2 x i64>, <2 x i64> } [[VLD2]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x2_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x2_t [[TMP6]]
poly64x2x2_t test_vld2q_p64(poly64_t const * ptr) {
  return vld2q_p64(ptr);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x3_t @test_vld3_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[VLD3:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld3.v1i64.p0(ptr %ptr)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 24, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x3_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x3_t [[TMP6]]
poly64x1x3_t test_vld3_p64(poly64_t const * ptr) {
  return vld3_p64(ptr);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x3_t @test_vld3q_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[VLD3:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld3.v2i64.p0(ptr %ptr)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64> } [[VLD3]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 48, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x3_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x3_t [[TMP6]]
poly64x2x3_t test_vld3q_p64(poly64_t const * ptr) {
  return vld3q_p64(ptr);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x1x4_t @test_vld4_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[VLD4:%.*]] = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.aarch64.neon.ld4.v1i64.p0(ptr %ptr)
// CHECK:   store { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[RETVAL]], ptr align 8 [[__RET]], i64 32, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x1x4_t, ptr [[RETVAL]], align 8
// CHECK:   ret %struct.poly64x1x4_t [[TMP6]]
poly64x1x4_t test_vld4_p64(poly64_t const * ptr) {
  return vld4_p64(ptr);
}

// CHECK-LABEL: define{{.*}} %struct.poly64x2x4_t @test_vld4q_p64(ptr noundef %ptr) #2 {
// CHECK:   [[RETVAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__RET:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[VLD4:%.*]] = call { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } @llvm.aarch64.neon.ld4.v2i64.p0(ptr %ptr)
// CHECK:   store { <2 x i64>, <2 x i64>, <2 x i64>, <2 x i64> } [[VLD4]], ptr [[__RET]]
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[RETVAL]], ptr align 16 [[__RET]], i64 64, i1 false)
// CHECK:   [[TMP6:%.*]] = load %struct.poly64x2x4_t, ptr [[RETVAL]], align 16
// CHECK:   ret %struct.poly64x2x4_t [[TMP6]]
poly64x2x4_t test_vld4q_p64(poly64_t const * ptr) {
  return vld4q_p64(ptr);
}

// CHECK-LABEL: define{{.*}} void @test_vst2_p64(ptr noundef %ptr, [2 x <1 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [2 x <1 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[VAL]], i64 16, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x1x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [2 x <1 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX3]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v1i64.p0(<1 x i64> [[TMP7]], <1 x i64> [[TMP8]], ptr %ptr)
// CHECK:   ret void
void test_vst2_p64(poly64_t * ptr, poly64x1x2_t val) {
  return vst2_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst2q_p64(ptr noundef %ptr, [2 x <2 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [2 x <2 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[VAL]], i64 32, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x2x2_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [2 x <2 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX3]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[TMP7:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP8:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st2.v2i64.p0(<2 x i64> [[TMP7]], <2 x i64> [[TMP8]], ptr %ptr)
// CHECK:   ret void
void test_vst2q_p64(poly64_t * ptr, poly64x2x2_t val) {
  return vst2q_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst3_p64(ptr noundef %ptr, [3 x <1 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [3 x <1 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[VAL]], i64 24, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX3]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL4:%.*]] = getelementptr inbounds %struct.poly64x1x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX5:%.*]] = getelementptr inbounds [3 x <1 x i64>], ptr [[VAL4]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX5]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v1i64.p0(<1 x i64> [[TMP9]], <1 x i64> [[TMP10]], <1 x i64> [[TMP11]], ptr %ptr)
// CHECK:   ret void
void test_vst3_p64(poly64_t * ptr, poly64x1x3_t val) {
  return vst3_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst3q_p64(ptr noundef %ptr, [3 x <2 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [3 x <2 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[VAL]], i64 48, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX3]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL4:%.*]] = getelementptr inbounds %struct.poly64x2x3_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX5:%.*]] = getelementptr inbounds [3 x <2 x i64>], ptr [[VAL4]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX5]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[TMP9:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP10:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st3.v2i64.p0(<2 x i64> [[TMP9]], <2 x i64> [[TMP10]], <2 x i64> [[TMP11]], ptr %ptr)
// CHECK:   ret void
void test_vst3q_p64(poly64_t * ptr, poly64x2x3_t val) {
  return vst3q_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst4_p64(ptr noundef %ptr, [4 x <1 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x1x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [4 x <1 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 8
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[__S1]], ptr align 8 [[VAL]], i64 32, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <1 x i64>, ptr [[ARRAYIDX]], align 8
// CHECK:   [[TMP4:%.*]] = bitcast <1 x i64> [[TMP3]] to <8 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <1 x i64>, ptr [[ARRAYIDX3]], align 8
// CHECK:   [[TMP6:%.*]] = bitcast <1 x i64> [[TMP5]] to <8 x i8>
// CHECK:   [[VAL4:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX5:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL4]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <1 x i64>, ptr [[ARRAYIDX5]], align 8
// CHECK:   [[TMP8:%.*]] = bitcast <1 x i64> [[TMP7]] to <8 x i8>
// CHECK:   [[VAL6:%.*]] = getelementptr inbounds %struct.poly64x1x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX7:%.*]] = getelementptr inbounds [4 x <1 x i64>], ptr [[VAL6]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <1 x i64>, ptr [[ARRAYIDX7]], align 8
// CHECK:   [[TMP10:%.*]] = bitcast <1 x i64> [[TMP9]] to <8 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <8 x i8> [[TMP4]] to <1 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <8 x i8> [[TMP6]] to <1 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <8 x i8> [[TMP8]] to <1 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <8 x i8> [[TMP10]] to <1 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v1i64.p0(<1 x i64> [[TMP11]], <1 x i64> [[TMP12]], <1 x i64> [[TMP13]], <1 x i64> [[TMP14]], ptr %ptr)
// CHECK:   ret void
void test_vst4_p64(poly64_t * ptr, poly64x1x4_t val) {
  return vst4_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} void @test_vst4q_p64(ptr noundef %ptr, [4 x <2 x i64>] %val.coerce) #2 {
// CHECK:   [[VAL:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[__S1:%.*]] = alloca %struct.poly64x2x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[VAL]], i32 0, i32 0
// CHECK:   store [4 x <2 x i64>] [[VAL]].coerce, ptr [[COERCE_DIVE]], align 16
// CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[__S1]], ptr align 16 [[VAL]], i64 64, i1 false)
// CHECK:   [[VAL1:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL1]], i64 0, i64 0
// CHECK:   [[TMP3:%.*]] = load <2 x i64>, ptr [[ARRAYIDX]], align 16
// CHECK:   [[TMP4:%.*]] = bitcast <2 x i64> [[TMP3]] to <16 x i8>
// CHECK:   [[VAL2:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX3:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL2]], i64 0, i64 1
// CHECK:   [[TMP5:%.*]] = load <2 x i64>, ptr [[ARRAYIDX3]], align 16
// CHECK:   [[TMP6:%.*]] = bitcast <2 x i64> [[TMP5]] to <16 x i8>
// CHECK:   [[VAL4:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX5:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL4]], i64 0, i64 2
// CHECK:   [[TMP7:%.*]] = load <2 x i64>, ptr [[ARRAYIDX5]], align 16
// CHECK:   [[TMP8:%.*]] = bitcast <2 x i64> [[TMP7]] to <16 x i8>
// CHECK:   [[VAL6:%.*]] = getelementptr inbounds %struct.poly64x2x4_t, ptr [[__S1]], i32 0, i32 0
// CHECK:   [[ARRAYIDX7:%.*]] = getelementptr inbounds [4 x <2 x i64>], ptr [[VAL6]], i64 0, i64 3
// CHECK:   [[TMP9:%.*]] = load <2 x i64>, ptr [[ARRAYIDX7]], align 16
// CHECK:   [[TMP10:%.*]] = bitcast <2 x i64> [[TMP9]] to <16 x i8>
// CHECK:   [[TMP11:%.*]] = bitcast <16 x i8> [[TMP4]] to <2 x i64>
// CHECK:   [[TMP12:%.*]] = bitcast <16 x i8> [[TMP6]] to <2 x i64>
// CHECK:   [[TMP13:%.*]] = bitcast <16 x i8> [[TMP8]] to <2 x i64>
// CHECK:   [[TMP14:%.*]] = bitcast <16 x i8> [[TMP10]] to <2 x i64>
// CHECK:   call void @llvm.aarch64.neon.st4.v2i64.p0(<2 x i64> [[TMP11]], <2 x i64> [[TMP12]], <2 x i64> [[TMP13]], <2 x i64> [[TMP14]], ptr %ptr)
// CHECK:   ret void
void test_vst4q_p64(poly64_t * ptr, poly64x2x4_t val) {
  return vst4q_p64(ptr, val);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vext_p64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <1 x i64> [[TMP2]], <1 x i64> [[TMP3]], <1 x i32> zeroinitializer
// CHECK:   ret <1 x i64> [[VEXT]]
poly64x1_t test_vext_p64(poly64x1_t a, poly64x1_t b) {
  return vext_u64(a, b, 0);

}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vextq_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[TMP2:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[TMP3:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VEXT:%.*]] = shufflevector <2 x i64> [[TMP2]], <2 x i64> [[TMP3]], <2 x i32> <i32 1, i32 2>
// CHECK:   ret <2 x i64> [[VEXT]]
poly64x2_t test_vextq_p64(poly64x2_t a, poly64x2_t b) {
  return vextq_p64(a, b, 1);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vzip1q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vzip1q_p64(poly64x2_t a, poly64x2_t b) {
  return vzip1q_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vzip2q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vzip2q_p64(poly64x2_t a, poly64x2_t b) {
  return vzip2q_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vuzp1q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vuzp1q_p64(poly64x2_t a, poly64x2_t b) {
  return vuzp1q_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vuzp2q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vuzp2q_p64(poly64x2_t a, poly64x2_t b) {
  return vuzp2q_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vtrn1q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 0, i32 2>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vtrn1q_p64(poly64x2_t a, poly64x2_t b) {
  return vtrn1q_p64(a, b);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vtrn2q_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[SHUFFLE_I:%.*]] = shufflevector <2 x i64> %a, <2 x i64> %b, <2 x i32> <i32 1, i32 3>
// CHECK:   ret <2 x i64> [[SHUFFLE_I]]
poly64x2_t test_vtrn2q_p64(poly64x2_t a, poly64x2_t b) {
  return vtrn2q_u64(a, b);
}

// CHECK-LABEL: define{{.*}} <1 x i64> @test_vsri_n_p64(<1 x i64> noundef %a, <1 x i64> noundef %b) #0 {
// CHECK:   [[TMP0:%.*]] = bitcast <1 x i64> %a to <8 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <1 x i64> %b to <8 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <8 x i8> [[TMP0]] to <1 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <8 x i8> [[TMP1]] to <1 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <1 x i64> @llvm.aarch64.neon.vsri.v1i64(<1 x i64> [[VSRI_N]], <1 x i64> [[VSRI_N1]], i32 33)
// CHECK:   ret <1 x i64> [[VSRI_N2]]
poly64x1_t test_vsri_n_p64(poly64x1_t a, poly64x1_t b) {
  return vsri_n_p64(a, b, 33);
}

// CHECK-LABEL: define{{.*}} <2 x i64> @test_vsriq_n_p64(<2 x i64> noundef %a, <2 x i64> noundef %b) #1 {
// CHECK:   [[TMP0:%.*]] = bitcast <2 x i64> %a to <16 x i8>
// CHECK:   [[TMP1:%.*]] = bitcast <2 x i64> %b to <16 x i8>
// CHECK:   [[VSRI_N:%.*]] = bitcast <16 x i8> [[TMP0]] to <2 x i64>
// CHECK:   [[VSRI_N1:%.*]] = bitcast <16 x i8> [[TMP1]] to <2 x i64>
// CHECK:   [[VSRI_N2:%.*]] = call <2 x i64> @llvm.aarch64.neon.vsri.v2i64(<2 x i64> [[VSRI_N]], <2 x i64> [[VSRI_N1]], i32 64)
// CHECK:   ret <2 x i64> [[VSRI_N2]]
poly64x2_t test_vsriq_n_p64(poly64x2_t a, poly64x2_t b) {
  return vsriq_n_p64(a, b, 64);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"
// CHECK: attributes #2 ={{.*}}"min-legal-vector-width"="0"
