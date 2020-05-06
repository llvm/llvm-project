// RUN: %clang_cc1 -ffreestanding  -triple=x86_64-apple-darwin -target-cpu skx %s -emit-llvm -o - | FileCheck %s
#include <xmmintrin.h>
// This test is complimented by the .ll test under llvm/test/MC/X86/. 
// At this level we can only check if the constarints are passed correctly
// from inline asm to llvm IR.

// CHECK-LABEL: @f_Ym
void f_Ym(__m64 m)
  {
  // CHECK: movq  $0, %mm1
  // CHECK-SAME: "=^Ym,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("movq %0, %%mm1\n\t"
          :"=Ym" (m));
}

// CHECK-LABEL: f_Yi
void f_Yi(__m128 x, __m128 y, __m128 z)
  {
  // CHECK: vpaddq
  // CHECK-SAME: "=^Yi,^Yi,^Yi,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("vpaddq %0, %1, %2\n\t"
          :"=Yi" (x)
          :"Yi" (y),"Yi"(z));
}

// CHECK-LABEL: f_Yt
void f_Yt(__m128 x, __m128 y, __m128 z)
  {
  // CHECK: vpaddq
  // CHECK-SAME: "=^Yt,^Yt,^Yt,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("vpaddq %0, %1, %2\n\t"
          :"=Yt" (x)
          :"Yt" (y),"Yt"(z));
}

// CHECK-LABEL: f_Y2
void f_Y2(__m128 x, __m128 y, __m128 z)
    {
  // CHECK: vpaddq
  // CHECK-SAME: "=^Y2,^Y2,^Y2,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("vpaddq %0, %1, %2\n\t"
            :"=Y2" (x)
            :"Y2" (y),"Y2"(z));
}

// CHECK-LABEL: f_Yz
void f_Yz(__m128 x, __m128 y, __m128 z)
  {
  // CHECK: vpaddq
  // CHECK-SAME: vpaddq
  // CHECK-SAME: "=^Yi,=^Yz,^Yi,0,~{dirflag},~{fpsr},~{flags}"
  __asm__ volatile ("vpaddq %0,%2,%1\n\t"
       "vpaddq %1,%0,%2\n\t"
          :"+Yi"(z),"=Yz" (x)
          :"Yi" (y) );
}
