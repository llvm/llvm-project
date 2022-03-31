// RUN: %clang_cc1 -triple arm64-none-linux-gnu -target-feature +neon \
// RUN: -disable-O0-optnone  -emit-llvm -o - %s | opt -S -mem2reg | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl1_s8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL11_I]]
int8x8_t test_vtbl1_s8(int8x8_t a, int8x8_t b) {
  return vtbl1_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl1_s8(<16 x i8> noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> %a, <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL1_I]]
int8x8_t test_vqtbl1_s8(int8x16_t a, uint8x8_t b) {
  return vqtbl1_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl2_s8([2 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[A]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL13_I]]
int8x8_t test_vtbl2_s8(int8x8x2_t a, int8x8_t b) {
  return vtbl2_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl2_s8([2 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL2_I]]
int8x8_t test_vqtbl2_s8(int8x16x2_t a, uint8x8_t b) {
  return vqtbl2_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl3_s8([3 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[A]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL26_I]]
int8x8_t test_vtbl3_s8(int8x8x3_t a, int8x8_t b) {
  return vtbl3_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl3_s8([3 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl3.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL3_I]]
int8x8_t test_vqtbl3_s8(int8x16x3_t a, uint8x8_t b) {
  return vqtbl3_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl4_s8([4 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[A]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL27_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL28_I]]
int8x8_t test_vtbl4_s8(int8x8x4_t a, int8x8_t b) {
  return vtbl4_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl4_s8([4 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl4.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL4_I]]
int8x8_t test_vqtbl4_s8(int8x16x4_t a, uint8x8_t b) {
  return vqtbl4_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl1q_s8(<16 x i8> noundef %a, <16 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> %a, <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL1_I]]
int8x16_t test_vqtbl1q_s8(int8x16_t a, int8x16_t b) {
  return vqtbl1q_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl2q_s8([2 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL2_I]]
int8x16_t test_vqtbl2q_s8(int8x16x2_t a, int8x16_t b) {
  return vqtbl2q_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl3q_s8([3 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL3_I]]
int8x16_t test_vqtbl3q_s8(int8x16x3_t a, int8x16_t b) {
  return vqtbl3q_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl4q_s8([4 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL4_I]]
int8x16_t test_vqtbl4q_s8(int8x16x4_t a, int8x16_t b) {
  return vqtbl4q_s8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx1_s8(<8 x i8> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %c) #3
// CHECK:   [[TMP0:%.*]] = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
// CHECK:   [[TMP1:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i8> [[TMP1]], %a
// CHECK:   [[TMP3:%.*]] = xor <8 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP4:%.*]] = and <8 x i8> [[TMP3]], [[VTBL11_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP2]], [[TMP4]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
int8x8_t test_vtbx1_s8(int8x8_t a, int8x8_t b, int8x8_t c) {
  return vtbx1_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx2_s8(<8 x i8> noundef %a, [2 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x2_t, %struct.int8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBX1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> [[VTBX1_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX13_I]]
int8x8_t test_vtbx2_s8(int8x8_t a, int8x8x2_t b, int8x8_t c) {
  return vtbx2_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx3_s8(<8 x i8> noundef %a, [3 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x8x3_t, %struct.int8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %c) #3
// CHECK:   [[TMP4:%.*]] = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
// CHECK:   [[TMP5:%.*]] = sext <8 x i1> [[TMP4]] to <8 x i8>
// CHECK:   [[TMP6:%.*]] = and <8 x i8> [[TMP5]], %a
// CHECK:   [[TMP7:%.*]] = xor <8 x i8> [[TMP5]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP8:%.*]] = and <8 x i8> [[TMP7]], [[VTBL26_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP6]], [[TMP8]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
int8x8_t test_vtbx3_s8(int8x8_t a, int8x8x3_t b, int8x8_t c) {
  return vtbx3_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx4_s8(<8 x i8> noundef %a, [4 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.int8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x8x4_t, %struct.int8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBX2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[VTBX2_I]], <16 x i8> [[VTBX27_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX28_I]]
int8x8_t test_vtbx4_s8(int8x8_t a, int8x8x4_t b, int8x8_t c) {
  return vtbx4_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx1_s8(<8 x i8> noundef %a, <16 x i8> noundef %b, <8 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX1_I]]
int8x8_t test_vqtbx1_s8(int8x8_t a, int8x16_t b, uint8x8_t c) {
  return vqtbx1_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx2_s8(<8 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX2_I]]
int8x8_t test_vqtbx2_s8(int8x8_t a, int8x16x2_t b, uint8x8_t c) {
  return vqtbx2_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx3_s8(<8 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx3.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX3_I]]
int8x8_t test_vqtbx3_s8(int8x8_t a, int8x16x3_t b, uint8x8_t c) {
  return vqtbx3_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx4_s8(<8 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx4.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX4_I]]
int8x8_t test_vqtbx4_s8(int8x8_t a, int8x16x4_t b, uint8x8_t c) {
  return vqtbx4_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx1q_s8(<16 x i8> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX1_I]]
int8x16_t test_vqtbx1q_s8(int8x16_t a, int8x16_t b, uint8x16_t c) {
  return vqtbx1q_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx2q_s8(<16 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x2_t, %struct.int8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx2.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX2_I]]
int8x16_t test_vqtbx2q_s8(int8x16_t a, int8x16x2_t b, int8x16_t c) {
  return vqtbx2q_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx3q_s8(<16 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x3_t, %struct.int8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx3.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX3_I]]
int8x16_t test_vqtbx3q_s8(int8x16_t a, int8x16x3_t b, int8x16_t c) {
  return vqtbx3q_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx4q_s8(<16 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.int8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.int8x16x4_t, %struct.int8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx4.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX4_I]]
int8x16_t test_vqtbx4q_s8(int8x16_t a, int8x16x4_t b, int8x16_t c) {
  return vqtbx4q_s8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl1_u8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL11_I]]
uint8x8_t test_vtbl1_u8(uint8x8_t a, uint8x8_t b) {
  return vtbl1_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl1_u8(<16 x i8> noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> %a, <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL1_I]]
uint8x8_t test_vqtbl1_u8(uint8x16_t a, uint8x8_t b) {
  return vqtbl1_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl2_u8([2 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[A]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL13_I]]
uint8x8_t test_vtbl2_u8(uint8x8x2_t a, uint8x8_t b) {
  return vtbl2_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl2_u8([2 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL2_I]]
uint8x8_t test_vqtbl2_u8(uint8x16x2_t a, uint8x8_t b) {
  return vqtbl2_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl3_u8([3 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[A]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL26_I]]
uint8x8_t test_vtbl3_u8(uint8x8x3_t a, uint8x8_t b) {
  return vtbl3_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl3_u8([3 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl3.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL3_I]]
uint8x8_t test_vqtbl3_u8(uint8x16x3_t a, uint8x8_t b) {
  return vqtbl3_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl4_u8([4 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[A]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL27_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL28_I]]
uint8x8_t test_vtbl4_u8(uint8x8x4_t a, uint8x8_t b) {
  return vtbl4_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl4_u8([4 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl4.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL4_I]]
uint8x8_t test_vqtbl4_u8(uint8x16x4_t a, uint8x8_t b) {
  return vqtbl4_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl1q_u8(<16 x i8> noundef %a, <16 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> %a, <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL1_I]]
uint8x16_t test_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
  return vqtbl1q_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl2q_u8([2 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL2_I]]
uint8x16_t test_vqtbl2q_u8(uint8x16x2_t a, uint8x16_t b) {
  return vqtbl2q_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl3q_u8([3 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL3_I]]
uint8x16_t test_vqtbl3q_u8(uint8x16x3_t a, uint8x16_t b) {
  return vqtbl3q_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl4q_u8([4 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL4_I]]
uint8x16_t test_vqtbl4q_u8(uint8x16x4_t a, uint8x16_t b) {
  return vqtbl4q_u8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx1_u8(<8 x i8> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %c) #3
// CHECK:   [[TMP0:%.*]] = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
// CHECK:   [[TMP1:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i8> [[TMP1]], %a
// CHECK:   [[TMP3:%.*]] = xor <8 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP4:%.*]] = and <8 x i8> [[TMP3]], [[VTBL11_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP2]], [[TMP4]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
uint8x8_t test_vtbx1_u8(uint8x8_t a, uint8x8_t b, uint8x8_t c) {
  return vtbx1_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx2_u8(<8 x i8> noundef %a, [2 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x2_t, %struct.uint8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBX1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> [[VTBX1_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX13_I]]
uint8x8_t test_vtbx2_u8(uint8x8_t a, uint8x8x2_t b, uint8x8_t c) {
  return vtbx2_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx3_u8(<8 x i8> noundef %a, [3 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x8x3_t, %struct.uint8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %c) #3
// CHECK:   [[TMP4:%.*]] = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
// CHECK:   [[TMP5:%.*]] = sext <8 x i1> [[TMP4]] to <8 x i8>
// CHECK:   [[TMP6:%.*]] = and <8 x i8> [[TMP5]], %a
// CHECK:   [[TMP7:%.*]] = xor <8 x i8> [[TMP5]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP8:%.*]] = and <8 x i8> [[TMP7]], [[VTBL26_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP6]], [[TMP8]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
uint8x8_t test_vtbx3_u8(uint8x8_t a, uint8x8x3_t b, uint8x8_t c) {
  return vtbx3_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx4_u8(<8 x i8> noundef %a, [4 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.uint8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x8x4_t, %struct.uint8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBX2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[VTBX2_I]], <16 x i8> [[VTBX27_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX28_I]]
uint8x8_t test_vtbx4_u8(uint8x8_t a, uint8x8x4_t b, uint8x8_t c) {
  return vtbx4_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx1_u8(<8 x i8> noundef %a, <16 x i8> noundef %b, <8 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX1_I]]
uint8x8_t test_vqtbx1_u8(uint8x8_t a, uint8x16_t b, uint8x8_t c) {
  return vqtbx1_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx2_u8(<8 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX2_I]]
uint8x8_t test_vqtbx2_u8(uint8x8_t a, uint8x16x2_t b, uint8x8_t c) {
  return vqtbx2_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx3_u8(<8 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx3.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX3_I]]
uint8x8_t test_vqtbx3_u8(uint8x8_t a, uint8x16x3_t b, uint8x8_t c) {
  return vqtbx3_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx4_u8(<8 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx4.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX4_I]]
uint8x8_t test_vqtbx4_u8(uint8x8_t a, uint8x16x4_t b, uint8x8_t c) {
  return vqtbx4_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx1q_u8(<16 x i8> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX1_I]]
uint8x16_t test_vqtbx1q_u8(uint8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vqtbx1q_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx2q_u8(<16 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x2_t, %struct.uint8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx2.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX2_I]]
uint8x16_t test_vqtbx2q_u8(uint8x16_t a, uint8x16x2_t b, uint8x16_t c) {
  return vqtbx2q_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx3q_u8(<16 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x3_t, %struct.uint8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx3.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX3_I]]
uint8x16_t test_vqtbx3q_u8(uint8x16_t a, uint8x16x3_t b, uint8x16_t c) {
  return vqtbx3q_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx4q_u8(<16 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.uint8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.uint8x16x4_t, %struct.uint8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx4.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX4_I]]
uint8x16_t test_vqtbx4q_u8(uint8x16_t a, uint8x16x4_t b, uint8x16_t c) {
  return vqtbx4q_u8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl1_p8(<8 x i8> noundef %a, <8 x i8> noundef %b) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %a, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL11_I]]
poly8x8_t test_vtbl1_p8(poly8x8_t a, uint8x8_t b) {
  return vtbl1_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl1_p8(<16 x i8> noundef %a, <8 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> %a, <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL1_I]]
poly8x8_t test_vqtbl1_p8(poly8x16_t a, uint8x8_t b) {
  return vqtbl1_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl2_p8([2 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[A]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL13_I]]
poly8x8_t test_vtbl2_p8(poly8x8x2_t a, uint8x8_t b) {
  return vtbl2_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl2_p8([2 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL2_I]]
poly8x8_t test_vqtbl2_p8(poly8x16x2_t a, uint8x8_t b) {
  return vqtbl2_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl3_p8([3 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[A]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL26_I]]
poly8x8_t test_vtbl3_p8(poly8x8x3_t a, uint8x8_t b) {
  return vtbl3_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl3_p8([3 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl3.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL3_I]]
poly8x8_t test_vqtbl3_p8(poly8x16x3_t a, uint8x8_t b) {
  return vqtbl3_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbl4_p8([4 x <8 x i8>] %a.coerce, <8 x i8> noundef %b) #0 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[A:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[A]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL27_I]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL28_I]]
poly8x8_t test_vtbl4_p8(poly8x8x4_t a, uint8x8_t b) {
  return vtbl4_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbl4_p8([4 x <16 x i8>] %a.coerce, <8 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl4.v8i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %b) #3
// CHECK:   ret <8 x i8> [[VTBL4_I]]
poly8x8_t test_vqtbl4_p8(poly8x16x4_t a, uint8x8_t b) {
  return vqtbl4_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl1q_p8(<16 x i8> noundef %a, <16 x i8> noundef %b) #1 {
// CHECK:   [[VTBL1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl1.v16i8(<16 x i8> %a, <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL1_I]]
poly8x16_t test_vqtbl1q_p8(poly8x16_t a, uint8x16_t b) {
  return vqtbl1q_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl2q_p8([2 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[A]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBL2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl2.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL2_I]]
poly8x16_t test_vqtbl2q_p8(poly8x16x2_t a, uint8x16_t b) {
  return vqtbl2q_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl3q_p8([3 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[A]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBL3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl3.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL3_I]]
poly8x16_t test_vqtbl3q_p8(poly8x16x3_t a, uint8x16_t b) {
  return vqtbl3q_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbl4q_p8([4 x <16 x i8>] %a.coerce, <16 x i8> noundef %b) #1 {
// CHECK:   [[__P0_I:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[A:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[A]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[A]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P0_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBL4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbl4.v16i8(<16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %b) #3
// CHECK:   ret <16 x i8> [[VTBL4_I]]
poly8x16_t test_vqtbl4q_p8(poly8x16x4_t a, uint8x16_t b) {
  return vqtbl4q_p8(a, b);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx1_p8(<8 x i8> noundef %a, <8 x i8> noundef %b, <8 x i8> noundef %c) #0 {
// CHECK:   [[VTBL1_I:%.*]] = shufflevector <8 x i8> %b, <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL11_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl1.v8i8(<16 x i8> [[VTBL1_I]], <8 x i8> %c) #3
// CHECK:   [[TMP0:%.*]] = icmp uge <8 x i8> %c, <i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8, i8 8>
// CHECK:   [[TMP1:%.*]] = sext <8 x i1> [[TMP0]] to <8 x i8>
// CHECK:   [[TMP2:%.*]] = and <8 x i8> [[TMP1]], %a
// CHECK:   [[TMP3:%.*]] = xor <8 x i8> [[TMP1]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP4:%.*]] = and <8 x i8> [[TMP3]], [[VTBL11_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP2]], [[TMP4]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
poly8x8_t test_vtbx1_p8(poly8x8_t a, poly8x8_t b, uint8x8_t c) {
  return vtbx1_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx2_p8(<8 x i8> noundef %a, [2 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x2_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[B]].coerce, [2 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <8 x i8>], [2 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <8 x i8>] [[TMP0]], [2 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x2_t, %struct.poly8x8x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <8 x i8>], [2 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VTBX1_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX13_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> [[VTBX1_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX13_I]]
poly8x8_t test_vtbx2_p8(poly8x8_t a, poly8x8x2_t b, uint8x8_t c) {
  return vtbx2_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx3_p8(<8 x i8> noundef %a, [3 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x3_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[B]].coerce, [3 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <8 x i8>], [3 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <8 x i8>] [[TMP0]], [3 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x8x3_t, %struct.poly8x8x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <8 x i8>], [3 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VTBL2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL25_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> zeroinitializer, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBL26_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbl2.v8i8(<16 x i8> [[VTBL2_I]], <16 x i8> [[VTBL25_I]], <8 x i8> %c) #3
// CHECK:   [[TMP4:%.*]] = icmp uge <8 x i8> %c, <i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24, i8 24>
// CHECK:   [[TMP5:%.*]] = sext <8 x i1> [[TMP4]] to <8 x i8>
// CHECK:   [[TMP6:%.*]] = and <8 x i8> [[TMP5]], %a
// CHECK:   [[TMP7:%.*]] = xor <8 x i8> [[TMP5]], <i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1>
// CHECK:   [[TMP8:%.*]] = and <8 x i8> [[TMP7]], [[VTBL26_I]]
// CHECK:   [[VTBX_I:%.*]] = or <8 x i8> [[TMP6]], [[TMP8]]
// CHECK:   ret <8 x i8> [[VTBX_I]]
poly8x8_t test_vtbx3_p8(poly8x8_t a, poly8x8x3_t b, uint8x8_t c) {
  return vtbx3_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vtbx4_p8(<8 x i8> noundef %a, [4 x <8 x i8>] %b.coerce, <8 x i8> noundef %c) #0 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[B:%.*]] = alloca %struct.poly8x8x4_t, align 8
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[B]].coerce, [4 x <8 x i8>]* [[COERCE_DIVE]], align 8
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <8 x i8>], [4 x <8 x i8>]* [[COERCE_DIVE1]], align 8
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <8 x i8>] [[TMP0]], [4 x <8 x i8>]* [[COERCE_DIVE_I]], align 8
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX_I]], align 8
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX2_I]], align 8
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX4_I]], align 8
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x8x4_t, %struct.poly8x8x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <8 x i8>], [4 x <8 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <8 x i8>, <8 x i8>* [[ARRAYIDX6_I]], align 8
// CHECK:   [[VTBX2_I:%.*]] = shufflevector <8 x i8> [[TMP1]], <8 x i8> [[TMP2]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX27_I:%.*]] = shufflevector <8 x i8> [[TMP3]], <8 x i8> [[TMP4]], <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
// CHECK:   [[VTBX28_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[VTBX2_I]], <16 x i8> [[VTBX27_I]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX28_I]]
poly8x8_t test_vtbx4_p8(poly8x8_t a, poly8x8x4_t b, uint8x8_t c) {
  return vtbx4_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx1_p8(<8 x i8> noundef %a, <16 x i8> noundef %b, <8 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx1.v8i8(<8 x i8> %a, <16 x i8> %b, <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX1_I]]
poly8x8_t test_vqtbx1_p8(poly8x8_t a, uint8x16_t b, uint8x8_t c) {
  return vqtbx1_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx2_p8(<8 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx2.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX2_I]]
poly8x8_t test_vqtbx2_p8(poly8x8_t a, poly8x16x2_t b, uint8x8_t c) {
  return vqtbx2_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx3_p8(<8 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx3.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX3_I]]
poly8x8_t test_vqtbx3_p8(poly8x8_t a, poly8x16x3_t b, uint8x8_t c) {
  return vqtbx3_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <8 x i8> @test_vqtbx4_p8(<8 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <8 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <8 x i8> @llvm.aarch64.neon.tbx4.v8i8(<8 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <8 x i8> %c) #3
// CHECK:   ret <8 x i8> [[VTBX4_I]]
poly8x8_t test_vqtbx4_p8(poly8x8_t a, poly8x16x4_t b, uint8x8_t c) {
  return vqtbx4_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx1q_p8(<16 x i8> noundef %a, <16 x i8> noundef %b, <16 x i8> noundef %c) #1 {
// CHECK:   [[VTBX1_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx1.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX1_I]]
poly8x16_t test_vqtbx1q_p8(poly8x16_t a, uint8x16_t b, uint8x16_t c) {
  return vqtbx1q_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx2q_p8(<16 x i8> noundef %a, [2 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x2_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[B]].coerce, [2 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [2 x <16 x i8>], [2 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [2 x <16 x i8>] [[TMP0]], [2 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x2_t, %struct.poly8x16x2_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [2 x <16 x i8>], [2 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VTBX2_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx2.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX2_I]]
poly8x16_t test_vqtbx2q_p8(poly8x16_t a, poly8x16x2_t b, uint8x16_t c) {
  return vqtbx2q_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx3q_p8(<16 x i8> noundef %a, [3 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x3_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[B]].coerce, [3 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [3 x <16 x i8>], [3 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [3 x <16 x i8>] [[TMP0]], [3 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x3_t, %struct.poly8x16x3_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [3 x <16 x i8>], [3 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VTBX3_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx3.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX3_I]]
poly8x16_t test_vqtbx3q_p8(poly8x16_t a, poly8x16x3_t b, uint8x16_t c) {
  return vqtbx3q_p8(a, b, c);
}

// CHECK-LABEL: define{{.*}} <16 x i8> @test_vqtbx4q_p8(<16 x i8> noundef %a, [4 x <16 x i8>] %b.coerce, <16 x i8> noundef %c) #1 {
// CHECK:   [[__P1_I:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[B:%.*]] = alloca %struct.poly8x16x4_t, align 16
// CHECK:   [[COERCE_DIVE:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[B]].coerce, [4 x <16 x i8>]* [[COERCE_DIVE]], align 16
// CHECK:   [[COERCE_DIVE1:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[B]], i32 0, i32 0
// CHECK:   [[TMP0:%.*]] = load [4 x <16 x i8>], [4 x <16 x i8>]* [[COERCE_DIVE1]], align 16
// CHECK:   [[COERCE_DIVE_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   store [4 x <16 x i8>] [[TMP0]], [4 x <16 x i8>]* [[COERCE_DIVE_I]], align 16
// CHECK:   [[VAL_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL_I]], i64 0, i64 0
// CHECK:   [[TMP1:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX_I]], align 16
// CHECK:   [[VAL1_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX2_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL1_I]], i64 0, i64 1
// CHECK:   [[TMP2:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX2_I]], align 16
// CHECK:   [[VAL3_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX4_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL3_I]], i64 0, i64 2
// CHECK:   [[TMP3:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX4_I]], align 16
// CHECK:   [[VAL5_I:%.*]] = getelementptr inbounds %struct.poly8x16x4_t, %struct.poly8x16x4_t* [[__P1_I]], i32 0, i32 0
// CHECK:   [[ARRAYIDX6_I:%.*]] = getelementptr inbounds [4 x <16 x i8>], [4 x <16 x i8>]* [[VAL5_I]], i64 0, i64 3
// CHECK:   [[TMP4:%.*]] = load <16 x i8>, <16 x i8>* [[ARRAYIDX6_I]], align 16
// CHECK:   [[VTBX4_I:%.*]] = call <16 x i8> @llvm.aarch64.neon.tbx4.v16i8(<16 x i8> %a, <16 x i8> [[TMP1]], <16 x i8> [[TMP2]], <16 x i8> [[TMP3]], <16 x i8> [[TMP4]], <16 x i8> %c) #3
// CHECK:   ret <16 x i8> [[VTBX4_I]]
poly8x16_t test_vqtbx4q_p8(poly8x16_t a, poly8x16x4_t b, uint8x16_t c) {
  return vqtbx4q_p8(a, b, c);
}

// CHECK: attributes #0 ={{.*}}"min-legal-vector-width"="64"
// CHECK: attributes #1 ={{.*}}"min-legal-vector-width"="128"
