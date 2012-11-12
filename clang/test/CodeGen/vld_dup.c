// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7a-linux-gnueabi \
// RUN:   -target-cpu cortex-a8 \
// RUN:   -emit-llvm -O0 -o - %s | FileCheck %s
#include <arm_neon.h>
int main(){
    int32_t v0[3];
    int32x2x3_t v1;
    int32_t v2[4];
    int32x2x4_t v3;
    int64x1x3_t v4;
    int64x1x4_t v5;
    int64_t v6[3];
    int64_t v7[4];

    v1 = vld3_dup_s32(v0);
// CHECK: [[T168:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld3lane.v2i32(i8* {{.*}}, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 {{[0-9]+}}, i32 {{[0-9]+}})
// CHECK-NEXT: [[T169:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T168]], 0
// CHECK-NEXT: [[T170:%.*]] = shufflevector <2 x i32> [[T169]], <2 x i32> [[T169]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T171:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T168]], <2 x i32> [[T170]], 0
// CHECK-NEXT: [[T172:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T171]], 1
// CHECK-NEXT: [[T173:%.*]] = shufflevector <2 x i32> [[T172]], <2 x i32> [[T172]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T174:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T171]], <2 x i32> [[T173]], 1
// CHECK-NEXT: [[T175:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T174]], 2
// CHECK-NEXT: [[T176:%.*]] = shufflevector <2 x i32> [[T175]], <2 x i32> [[T175]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T177:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32> } [[T174]], <2 x i32> [[T176]], 2

    v3 = vld4_dup_s32(v2);
// CHECK: [[T178:%.*]] = call { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } @llvm.arm.neon.vld4lane.v2i32(i8* {{.*}}, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, <2 x i32> undef, i32 {{[0-9]+}}, i32 {{[0-9]+}})
// CHECK-NEXT: [[T179:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T178]], 0
// CHECK-NEXT: [[T180:%.*]] = shufflevector <2 x i32> [[T179]], <2 x i32> [[T179]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T181:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T178]], <2 x i32> [[T180]], 0
// CHECK-NEXT: [[T182:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T181]], 1
// CHECK-NEXT: [[T183:%.*]] = shufflevector <2 x i32> [[T182]], <2 x i32> [[T182]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T184:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T181]], <2 x i32> [[T183]], 1
// CHECK-NEXT: [[T185:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T184]], 2
// CHECK-NEXT: [[T186:%.*]] = shufflevector <2 x i32> [[T185]], <2 x i32> [[T185]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T187:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T184]], <2 x i32> [[T186]], 2
// CHECK-NEXT: [[T188:%.*]] = extractvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T187]], 3
// CHECK-NEXT: [[T189:%.*]] = shufflevector <2 x i32> [[T188]], <2 x i32> [[T188]], <2 x i32> zeroinitializer
// CHECK-NEXT: [[T190:%.*]] = insertvalue { <2 x i32>, <2 x i32>, <2 x i32>, <2 x i32> } [[T187]], <2 x i32> [[T189]], 3

    v4 = vld3_dup_s64(v6);
// CHECK: {{%.*}} = call { <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld3.v1i64(i8* {{.*}}, i32 {{[0-9]+}})

    v5 = vld4_dup_s64(v7);
// CHECK: {{%.*}} = call { <1 x i64>, <1 x i64>, <1 x i64>, <1 x i64> } @llvm.arm.neon.vld4.v1i64(i8* {{.*}}, i32 {{[0-9]+}})

    return 0;
}
