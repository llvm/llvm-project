// RUN: %clang_cc1 -O3 -ffp-contract=fast -triple=aarch64-apple-darwin -S -o - %s | FileCheck --check-prefix=CHECK-FMADD %s
// REQUIRES: aarch64-registered-target

float fma_test1(float a, float b, float c) {
// CHECK-FMADD: fmadd
 float x = a * b;
 float y = x + c;
 return y;
}

// RUN: %clang_cc1 -triple=x86_64 %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-DEFAULT %s
//
// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=off %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-DEFAULT %s
// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=on %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-ON %s
// RUN: %clang_cc1 -triple=x86_64 -ffp-contract=fast %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-CONTRACTFAST %s
//
// RUN: %clang_cc1 -triple=x86_64 -ffast-math %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-DEFAULTFAST %s
// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=off %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-DEFAULTFAST %s
// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=on %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-ONFAST %s
// RUN: %clang_cc1 -triple=x86_64 -ffast-math -ffp-contract=fast %s -emit-llvm -o - \
// RUN:| FileCheck --check-prefix=CHECK-FASTFAST %s
float mymuladd( float x, float y, float z ) {
  return x * y + z;
  // CHECK-DEFAULT: = fmul float
  // CHECK-DEFAULT: = fadd float
  
  // CHECK-ON: = call float @llvm.fmuladd.f32
 
  // CHECK-CONTRACTFAST: = fmul contract float
  // CHECK-CONTRACTFAST: = fadd contract float

  // CHECK-DEFAULTFAST: = fmul reassoc nnan ninf nsz arcp afn float
  // CHECK-DEFAULTFAST: = fadd reassoc nnan ninf nsz arcp afn float

  // CHECK-ONFAST: = call reassoc nnan ninf nsz arcp afn float @llvm.fmuladd.f32

  // CHECK-FASTFAST: = fmul fast float
  // CHECK-FASTFAST: = fadd fast float
}
