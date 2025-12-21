// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -fmath-errno %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefixes=ALL,CIR-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t-no-errno.cir
// RUN: FileCheck --input-file=%t-no-errno.cir %s -check-prefixes=ALL,CIR-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -fmath-errno -O1 %s -o %t-errno-o1.cir
// RUN: FileCheck --input-file=%t-errno-o1.cir %s -check-prefixes=ALL,CIR-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir -ffast-math -O1 %s -o %t-no-errno-o1.cir
// RUN: FileCheck --input-file=%t-no-errno-o1.cir %s -check-prefixes=ALL,CIR-NO-ERRNO-O1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -fmath-errno %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=ALL,LLVM-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -ffast-math %s -o %t-no-errno.ll
// RUN: FileCheck --input-file=%t-no-errno.ll %s -check-prefixes=ALL,LLVM-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -fmath-errno -O1 %s -o %t-errno-o1.ll
// RUN: FileCheck --input-file=%t-errno-o1.ll %s -check-prefixes=ALL,LLVM-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm -ffast-math -O1 %s -o %t-no-errno-o1.ll
// RUN: FileCheck --input-file=%t-no-errno-o1.ll %s -check-prefixes=ALL,LLVM-NO-ERRNO-O1

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -fmath-errno %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefixes=ALL,OGCG-ERRNO
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -ffast-math %s -o %t-no-errno.ll
// RUN: FileCheck --input-file=%t-no-errno.ll %s -check-prefixes=ALL,OGCG-NO-ERRNO

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -fmath-errno -O1 %s -o %t-errno-o1.ll
// RUN: FileCheck --input-file=%t-errno-o1.ll %s -check-prefixes=ALL,OGCG-ERRNO-O1
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm -ffast-math -O1 %s -o %t-no-errno-o1.ll
// RUN: FileCheck --input-file=%t-no-errno-o1.ll %s -check-prefixes=ALL,OGCG-NO-ERRNO-O1

float test_normal(float f) {
  return __builtin_cosf(f);
  // ALL: test_normal
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
  // ALL: test_precise
  // CIR-ERRNO: cir.call @cosf
  // CIR-NO-ERRNO: cir.call @cosf
  // LLVM-ERRNO: call float @cosf
  // LLVM-NO-ERRNO: call float @cosf
  // OGCG-ERRNO: call {{.*}} float @cosf
  // OGCG-NO-ERRNO: call {{.*}} float @cosf
}

float test_fast(float f) {
#pragma float_control(precise, off)
  // Should produce an intrinsic at -O1
  return __builtin_cosf(f);
  // ALL: test_fast
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
  // ALL: test_optnone
  // CIR-ERRNO: cir.call @cosf
  // CIR-NO-ERRNO: cir.call @cosf
  // LLVM-ERRNO: call float @cosf
  // LLVM-NO-ERRNO: call float @cosf
  // OGCG-ERRNO: call {{.*}} float @cosf
  // OGCG-NO-ERRNO: call {{.*}} float @cosf
}
