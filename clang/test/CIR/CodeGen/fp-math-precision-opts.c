// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -fmath-errno %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -ffast-math %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -fmath-errno -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -ffast-math -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR-NO-ERRNO-O1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -fmath-errno %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -ffast-math %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -fmath-errno -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -ffast-math -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=LLVM-NO-ERRNO-O1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -fmath-errno %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -ffast-math %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -fmath-errno -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -ffast-math -O1 %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=OGCG-NO-ERRNO-O1

float test_normal(float f) {
  return __builtin_cosf(f);
  // CIR-ERRNO: cir.call @cosf
  // CIR-NO-ERRNO: cir.cos
  // LLVM-ERRNO: call float @cosf
  // LLVM-NO-ERRNO: call float @llvm.cos.f32
  // OGCG-ERRNO: call {{.*}} float @cosf
  // OGCG-NO-ERRNO: call {{.*}} float @llvm.cos.f32
}

float test_precise(float f) {
#pragma float_control(precise, on)
  // Should never produce an intrinsic
  return __builtin_cosf(f);
  // CIR-ERRNO: cir.call @cosf
  // CIR-NO-ERRNO: cir.call @cosf
  // LLVM-ERRNO: call float @cosf
  // LLVM-NO-ERRNO: call float @llvm.cos.f32
  // OGCG-ERRNO: call {{.*}} float @cosf
  // OGCG-NO-ERRNO: call {{.*}} float @llvm.cos.f32
}

float test_fast(float f) {
#pragma float_control(precise, off)
  // Should produce an intrinsic at -O1
  return __builtin_cosf(f);
  // CIR-ERRNO-O1: cir.cos
  // CIR-NO-ERRNO-O1: cir.cos
  // LLVM-ERRNO-O1: call float @llvm.cos.f32
  // LLVM-NO-ERRNO-O1: call float @llvm.cos.f32
  // OGCG-ERRNO-O1: call {{.*}} float @llvm.cos.f32
  // OGCG-NO-ERRNO-O1: call {{.*}} float @llvm.cos.f32
}

__attribute__((optnone))
float test_optnone(float f) {
  // Should never produce an intrinsic
  return __builtin_cosf(f);
  // CIR-ERRNO: cir.call @cosf
  // CIR-NO-ERRNO: cir.call @cosf
  // LLVM-ERRNO: call float @cosf
  // LLVM-NO-ERRNO: call float @cosf
  // OGCG-ERRNO: call {{.*}} float @cosf
  // OGCG-NO-ERRNO: call {{.*}} float @cosf
}
