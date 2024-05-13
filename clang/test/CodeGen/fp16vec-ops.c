// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple arm64-apple-ios9 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple armv7-apple-ios9 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -triple x86_64-apple-macos10.13 -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK

typedef __fp16 half4 __attribute__ ((vector_size (8)));
typedef short short4 __attribute__ ((vector_size (8)));

half4 hv0, hv1;
short4 sv0;

// CHECK-LABEL: testFP16Vec0
// CHECK: %[[V0:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV:.*]] = fpext <4 x half> %[[V0]] to <4 x float>
// CHECK: %[[V1:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV1:.*]] = fpext <4 x half> %[[V1]] to <4 x float>
// CHECK: %[[ADD:.*]] = fadd <4 x float> %[[CONV]], %[[CONV1]]
// CHECK: %[[CONV2:.*]] = fptrunc <4 x float> %[[ADD]] to <4 x half>
// CHECK: store <4 x half> %[[CONV2]], ptr @hv0, align 8
// CHECK: %[[V2:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV3:.*]] = fpext <4 x half> %[[V2]] to <4 x float>
// CHECK: %[[V3:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV4:.*]] = fpext <4 x half> %[[V3]] to <4 x float>
// CHECK: %[[SUB:.*]] = fsub <4 x float> %[[CONV3]], %[[CONV4]]
// CHECK: %[[CONV5:.*]] = fptrunc <4 x float> %[[SUB]] to <4 x half>
// CHECK: store <4 x half> %[[CONV5]], ptr @hv0, align 8
// CHECK: %[[V4:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV6:.*]] = fpext <4 x half> %[[V4]] to <4 x float>
// CHECK: %[[V5:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV7:.*]] = fpext <4 x half> %[[V5]] to <4 x float>
// CHECK: %[[MUL:.*]] = fmul <4 x float> %[[CONV6]], %[[CONV7]]
// CHECK: %[[CONV8:.*]] = fptrunc <4 x float> %[[MUL]] to <4 x half>
// CHECK: store <4 x half> %[[CONV8]], ptr @hv0, align 8
// CHECK: %[[V6:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV9:.*]] = fpext <4 x half> %[[V6]] to <4 x float>
// CHECK: %[[V7:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV10:.*]] = fpext <4 x half> %[[V7]] to <4 x float>
// CHECK: %[[DIV:.*]] = fdiv <4 x float> %[[CONV9]], %[[CONV10]]
// CHECK: %[[CONV11:.*]] = fptrunc <4 x float> %[[DIV]] to <4 x half>
// CHECK: store <4 x half> %[[CONV11]], ptr @hv0, align 8

void testFP16Vec0() {
  hv0 = hv0 + hv1;
  hv0 = hv0 - hv1;
  hv0 = hv0 * hv1;
  hv0 = hv0 / hv1;
}

// CHECK-LABEL: testFP16Vec1
// CHECK: %[[V0:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV:.*]] = fpext <4 x half> %[[V0]] to <4 x float>
// CHECK: %[[V1:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV1:.*]] = fpext <4 x half> %[[V1]] to <4 x float>
// CHECK: %[[ADD:.*]] = fadd <4 x float> %[[CONV1]], %[[CONV]]
// CHECK: %[[CONV2:.*]] = fptrunc <4 x float> %[[ADD]] to <4 x half>
// CHECK: store <4 x half> %[[CONV2]], ptr @hv0, align 8
// CHECK: %[[V2:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV3:.*]] = fpext <4 x half> %[[V2]] to <4 x float>
// CHECK: %[[V3:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV4:.*]] = fpext <4 x half> %[[V3]] to <4 x float>
// CHECK: %[[SUB:.*]] = fsub <4 x float> %[[CONV4]], %[[CONV3]]
// CHECK: %[[CONV5:.*]] = fptrunc <4 x float> %[[SUB]] to <4 x half>
// CHECK: store <4 x half> %[[CONV5]], ptr @hv0, align 8
// CHECK: %[[V4:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV6:.*]] = fpext <4 x half> %[[V4]] to <4 x float>
// CHECK: %[[V5:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV7:.*]] = fpext <4 x half> %[[V5]] to <4 x float>
// CHECK: %[[MUL:.*]] = fmul <4 x float> %[[CONV7]], %[[CONV6]]
// CHECK: %[[CONV8:.*]] = fptrunc <4 x float> %[[MUL]] to <4 x half>
// CHECK: store <4 x half> %[[CONV8]], ptr @hv0, align 8
// CHECK: %[[V6:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV9:.*]] = fpext <4 x half> %[[V6]] to <4 x float>
// CHECK: %[[V7:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV10:.*]] = fpext <4 x half> %[[V7]] to <4 x float>
// CHECK: %[[DIV:.*]] = fdiv <4 x float> %[[CONV10]], %[[CONV9]]
// CHECK: %[[CONV11:.*]] = fptrunc <4 x float> %[[DIV]] to <4 x half>
// CHECK: store <4 x half> %[[CONV11]], ptr @hv0, align 8

void testFP16Vec1() {
  hv0 += hv1;
  hv0 -= hv1;
  hv0 *= hv1;
  hv0 /= hv1;
}

// CHECK-LABEL: testFP16Vec2
// CHECK: %[[CADDR:.*]] = alloca i32, align 4
// CHECK: store i32 %[[C:.*]], ptr %[[CADDR]], align 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[CADDR]], align 4
// CHECK: %[[TOBOOL:.*]] = icmp ne i32 %[[V0]], 0
// CHECK: br i1 %[[TOBOOL]], label %{{.*}}, label %{{.*}}
//
// CHECK: %[[V1:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: br label %{{.*}}
//
// CHECK: %[[V2:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: br label %{{.*}}
//
// CHECK: %[[COND:.*]] = phi <4 x half> [ %[[V1]], %{{.*}} ], [ %[[V2]], %{{.*}} ]
// CHECK: store <4 x half> %[[COND]], ptr @hv0, align 8

void testFP16Vec2(int c) {
  hv0 = c ? hv0 : hv1;
}

// CHECK-LABEL: testFP16Vec3
// CHECK: %[[V0:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV:.*]] = fpext <4 x half> %[[V0]] to <4 x float>
// CHECK: %[[V1:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV1:.*]] = fpext <4 x half> %[[V1]] to <4 x float>
// CHECK: %[[CMP:.*]] = fcmp oeq <4 x float> %[[CONV]], %[[CONV1]]
// CHECK: %[[SEXT:.*]] = sext <4 x i1> %[[CMP]] to <4 x i32>
// CHECK: %[[CONV2:.*]] = trunc <4 x i32> %[[SEXT]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV2]], ptr @sv0, align 8
// CHECK: %[[V2:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV3:.*]] = fpext <4 x half> %[[V2]] to <4 x float>
// CHECK: %[[V3:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV4:.*]] = fpext <4 x half> %[[V3]] to <4 x float>
// CHECK: %[[CMP5:.*]] = fcmp une <4 x float> %[[CONV3]], %[[CONV4]]
// CHECK: %[[SEXT6:.*]] = sext <4 x i1> %[[CMP5]] to <4 x i32>
// CHECK: %[[CONV7:.*]] = trunc <4 x i32> %[[SEXT6]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV7]], ptr @sv0, align 8
// CHECK: %[[V4:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV8:.*]] = fpext <4 x half> %[[V4]] to <4 x float>
// CHECK: %[[V5:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV9:.*]] = fpext <4 x half> %[[V5]] to <4 x float>
// CHECK: %[[CMP10:.*]] = fcmp olt <4 x float> %[[CONV8]], %[[CONV9]]
// CHECK: %[[SEXT11:.*]] = sext <4 x i1> %[[CMP10]] to <4 x i32>
// CHECK: %[[CONV12:.*]] = trunc <4 x i32> %[[SEXT11]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV12]], ptr @sv0, align 8
// CHECK: %[[V6:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV13:.*]] = fpext <4 x half> %[[V6]] to <4 x float>
// CHECK: %[[V7:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV14:.*]] = fpext <4 x half> %[[V7]] to <4 x float>
// CHECK: %[[CMP15:.*]] = fcmp ogt <4 x float> %[[CONV13]], %[[CONV14]]
// CHECK: %[[SEXT16:.*]] = sext <4 x i1> %[[CMP15]] to <4 x i32>
// CHECK: %[[CONV17:.*]] = trunc <4 x i32> %[[SEXT16]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV17]], ptr @sv0, align 8
// CHECK: %[[V8:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV18:.*]] = fpext <4 x half> %[[V8]] to <4 x float>
// CHECK: %[[V9:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV19:.*]] = fpext <4 x half> %[[V9]] to <4 x float>
// CHECK: %[[CMP20:.*]] = fcmp ole <4 x float> %[[CONV18]], %[[CONV19]]
// CHECK: %[[SEXT21:.*]] = sext <4 x i1> %[[CMP20]] to <4 x i32>
// CHECK: %[[CONV22:.*]] = trunc <4 x i32> %[[SEXT21]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV22]], ptr @sv0, align 8
// CHECK: %[[V10:.*]] = load <4 x half>, ptr @hv0, align 8
// CHECK: %[[CONV23:.*]] = fpext <4 x half> %[[V10]] to <4 x float>
// CHECK: %[[V11:.*]] = load <4 x half>, ptr @hv1, align 8
// CHECK: %[[CONV24:.*]] = fpext <4 x half> %[[V11]] to <4 x float>
// CHECK: %[[CMP25:.*]] = fcmp oge <4 x float> %[[CONV23]], %[[CONV24]]
// CHECK: %[[SEXT26:.*]] = sext <4 x i1> %[[CMP25]] to <4 x i32>
// CHECK: %[[CONV27:.*]] = trunc <4 x i32> %[[SEXT26]] to <4 x i16>
// CHECK: store <4 x i16> %[[CONV27]], ptr @sv0, align 8

void testFP16Vec3() {
  sv0 = hv0 == hv1;
  sv0 = hv0 != hv1;
  sv0 = hv0 < hv1;
  sv0 = hv0 > hv1;
  sv0 = hv0 <= hv1;
  sv0 = hv0 >= hv1;
}
