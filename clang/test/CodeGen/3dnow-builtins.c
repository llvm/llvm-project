// REQUIRES: x86-registered-target
// RUN: %clang_cc1 %s -triple=x86_64-unknown-unknown -target-feature +3dnow -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-unknown-unknown -target-feature +3dnow -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <x86intrin.h>

__m64 test_m_pavgusb(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pavgusb
  // CHECK: @llvm.x86.3dnow.pavgusb
  // CHECK-ASM: pavgusb %mm{{.*}}, %mm{{.*}}
  return _m_pavgusb(m1, m2);
}

__m64 test_m_pf2id(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pf2id
  // CHECK: @llvm.x86.3dnow.pf2id
  // CHECK-ASM: pf2id %mm{{.*}}, %mm{{.*}}
  return _m_pf2id(m);
}

__m64 test_m_pfacc(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfacc
  // CHECK: @llvm.x86.3dnow.pfacc
  // CHECK-ASM: pfacc %mm{{.*}}, %mm{{.*}}
  return _m_pfacc(m1, m2);
}

__m64 test_m_pfadd(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfadd
  // CHECK: @llvm.x86.3dnow.pfadd
  // CHECK-ASM: pfadd %mm{{.*}}, %mm{{.*}}
  return _m_pfadd(m1, m2);
}

__m64 test_m_pfcmpeq(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfcmpeq
  // CHECK: @llvm.x86.3dnow.pfcmpeq
  // CHECK-ASM: pfcmpeq %mm{{.*}}, %mm{{.*}}
  return _m_pfcmpeq(m1, m2);
}

__m64 test_m_pfcmpge(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfcmpge
  // CHECK: @llvm.x86.3dnow.pfcmpge
  // CHECK-ASM: pfcmpge %mm{{.*}}, %mm{{.*}}
  return _m_pfcmpge(m1, m2);
}

__m64 test_m_pfcmpgt(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfcmpgt
  // CHECK: @llvm.x86.3dnow.pfcmpgt
  // CHECK-ASM: pfcmpgt %mm{{.*}}, %mm{{.*}}
  return _m_pfcmpgt(m1, m2);
}

__m64 test_m_pfmax(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfmax
  // CHECK: @llvm.x86.3dnow.pfmax
  // CHECK-ASM: pfmax %mm{{.*}}, %mm{{.*}}
  return _m_pfmax(m1, m2);
}

__m64 test_m_pfmin(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfmin
  // CHECK: @llvm.x86.3dnow.pfmin
  // CHECK-ASM: pfmin %mm{{.*}}, %mm{{.*}}
  return _m_pfmin(m1, m2);
}

__m64 test_m_pfmul(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfmul
  // CHECK: @llvm.x86.3dnow.pfmul
  // CHECK-ASM: pfmul %mm{{.*}}, %mm{{.*}}
  return _m_pfmul(m1, m2);
}

__m64 test_m_pfrcp(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pfrcp
  // CHECK: @llvm.x86.3dnow.pfrcp
  // CHECK-ASM: pfrcp %mm{{.*}}, %mm{{.*}}
  return _m_pfrcp(m);
}

__m64 test_m_pfrcpit1(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfrcpit1
  // CHECK: @llvm.x86.3dnow.pfrcpit1
  // CHECK-ASM: pfrcpit1 %mm{{.*}}, %mm{{.*}}
  return _m_pfrcpit1(m1, m2);
}

__m64 test_m_pfrcpit2(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfrcpit2
  // CHECK: @llvm.x86.3dnow.pfrcpit2
  // CHECK-ASM: pfrcpit2 %mm{{.*}}, %mm{{.*}}
  return _m_pfrcpit2(m1, m2);
}

__m64 test_m_pfrsqrt(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pfrsqrt
  // CHECK: @llvm.x86.3dnow.pfrsqrt
  // CHECK-ASM: pfrsqrt %mm{{.*}}, %mm{{.*}}
  return _m_pfrsqrt(m);
}

__m64 test_m_pfrsqrtit1(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfrsqrtit1
  // CHECK: @llvm.x86.3dnow.pfrsqit1
  // CHECK-ASM: pfrsqit1 %mm{{.*}}, %mm{{.*}}
  return _m_pfrsqrtit1(m1, m2);
}

__m64 test_m_pfsub(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfsub
  // CHECK: @llvm.x86.3dnow.pfsub
  // CHECK-ASM: pfsub %mm{{.*}}, %mm{{.*}}
  return _m_pfsub(m1, m2);
}

__m64 test_m_pfsubr(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfsubr
  // CHECK: @llvm.x86.3dnow.pfsubr
  // CHECK-ASM: pfsubr %mm{{.*}}, %mm{{.*}}
  return _m_pfsubr(m1, m2);
}

__m64 test_m_pi2fd(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pi2fd
  // CHECK: @llvm.x86.3dnow.pi2fd
  // CHECK-ASM: pi2fd %mm{{.*}}, %mm{{.*}}
  return _m_pi2fd(m);
}

__m64 test_m_pmulhrw(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pmulhrw
  // CHECK: @llvm.x86.3dnow.pmulhrw
  return _m_pmulhrw(m1, m2);
}

__m64 test_m_pf2iw(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pf2iw
  // CHECK: @llvm.x86.3dnowa.pf2iw
  // CHECK-ASM: pf2iw %mm{{.*}}, %mm{{.*}}
  return _m_pf2iw(m);
}

__m64 test_m_pfnacc(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfnacc
  // CHECK: @llvm.x86.3dnowa.pfnacc
  // CHECK-ASM: pfnacc %mm{{.*}}, %mm{{.*}}
  return _m_pfnacc(m1, m2);
}

__m64 test_m_pfpnacc(__m64 m1, __m64 m2) {
  // CHECK-LABEL: define i64 @test_m_pfpnacc
  // CHECK: @llvm.x86.3dnowa.pfpnacc
  // CHECK-ASM: pfpnacc %mm{{.*}}, %mm{{.*}}
  return _m_pfpnacc(m1, m2);
}

__m64 test_m_pi2fw(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pi2fw
  // CHECK: @llvm.x86.3dnowa.pi2fw
  // CHECK-ASM: pi2fw %mm{{.*}}, %mm{{.*}}
  return _m_pi2fw(m);
}

__m64 test_m_pswapdsf(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pswapdsf
  // CHECK: @llvm.x86.3dnowa.pswapd
  // CHECK-ASM: pswapd %mm{{.*}}, %mm{{.*}}
  return _m_pswapdsf(m);
}

__m64 test_m_pswapdsi(__m64 m) {
  // CHECK-LABEL: define i64 @test_m_pswapdsi
  // CHECK: @llvm.x86.3dnowa.pswapd
  // CHECK-ASM: pswapd %mm{{.*}}, %mm{{.*}}
  return _m_pswapdsi(m);
}
