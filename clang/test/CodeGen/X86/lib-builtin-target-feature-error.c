// RUN: %clang_cc1 %s -triple=i686-linux-gnu -target-feature -sse -emit-llvm -verify -o /dev/null
// RUN: %clang_cc1 %s -triple=i686-linux-gnu -target-feature +sse2 -emit-llvm -o - -DNOERROR | FileCheck %s

#ifdef NOERROR
// expected-no-diagnostics
#endif

#define __MM_MALLOC_H
#include <emmintrin.h>

// CHECK-LABEL: define{{.*}} void @test(
// CHECK: call void @llvm.x86.sse.sfence()
// CHECK: call void @llvm.x86.sse2.lfence()
// CHECK: call void @llvm.x86.sse2.mfence()
// CHECK: call void @llvm.x86.sse2.clflush(ptr {{.*}})
void test(void const *p) {
#ifndef NOERROR
  // expected-error@+5 {{'_mm_sfence' needs target feature sse}}
  // expected-error@+5 {{'_mm_lfence' needs target feature sse2}}
  // expected-error@+5 {{'_mm_mfence' needs target feature sse2}}
  // expected-error@+5 {{'_mm_clflush' needs target feature sse2}}
#endif
  _mm_sfence();
  _mm_lfence();
  _mm_mfence();
  _mm_clflush(p);
}
