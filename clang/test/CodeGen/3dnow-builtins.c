// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +3dnowa -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=GCC -check-prefix=CHECK
// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-scei-ps4 -target-feature +3dnowa -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=PS4 -check-prefix=CHECK


#include <x86intrin.h>

__m64 test_m_pavgusb(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pavgusb
  // GCC-LABEL: define double @test_m_pavgusb
  // CHECK: @llvm.x86.3dnow.pavgusb
  return _m_pavgusb(m1, m2);
}

__m64 test_m_pf2id(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pf2id
  // GCC-LABEL: define double @test_m_pf2id
  // CHECK: @llvm.x86.3dnow.pf2id
  return _m_pf2id(m);
}

__m64 test_m_pfacc(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfacc
  // GCC-LABEL: define double @test_m_pfacc
  // CHECK: @llvm.x86.3dnow.pfacc
  return _m_pfacc(m1, m2);
}

__m64 test_m_pfadd(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfadd
  // GCC-LABEL: define double @test_m_pfadd
  // CHECK: @llvm.x86.3dnow.pfadd
  return _m_pfadd(m1, m2);
}

__m64 test_m_pfcmpeq(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfcmpeq
  // GCC-LABEL: define double @test_m_pfcmpeq
  // CHECK: @llvm.x86.3dnow.pfcmpeq
  return _m_pfcmpeq(m1, m2);
}

__m64 test_m_pfcmpge(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfcmpge
  // GCC-LABEL: define double @test_m_pfcmpge
  // CHECK: @llvm.x86.3dnow.pfcmpge
  return _m_pfcmpge(m1, m2);
}

__m64 test_m_pfcmpgt(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfcmpgt
  // GCC-LABEL: define double @test_m_pfcmpgt
  // CHECK: @llvm.x86.3dnow.pfcmpgt
  return _m_pfcmpgt(m1, m2);
}

__m64 test_m_pfmax(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfmax
  // GCC-LABEL: define double @test_m_pfmax
  // CHECK: @llvm.x86.3dnow.pfmax
  return _m_pfmax(m1, m2);
}

__m64 test_m_pfmin(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfmin
  // GCC-LABEL: define double @test_m_pfmin
  // CHECK: @llvm.x86.3dnow.pfmin
  return _m_pfmin(m1, m2);
}

__m64 test_m_pfmul(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfmul
  // GCC-LABEL: define double @test_m_pfmul
  // CHECK: @llvm.x86.3dnow.pfmul
  return _m_pfmul(m1, m2);
}

__m64 test_m_pfrcp(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pfrcp
  // GCC-LABEL: define double @test_m_pfrcp
  // CHECK: @llvm.x86.3dnow.pfrcp
  return _m_pfrcp(m);
}

__m64 test_m_pfrcpit1(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfrcpit1
  // GCC-LABEL: define double @test_m_pfrcpit1
  // CHECK: @llvm.x86.3dnow.pfrcpit1
  return _m_pfrcpit1(m1, m2);
}

__m64 test_m_pfrcpit2(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfrcpit2
  // GCC-LABEL: define double @test_m_pfrcpit2
  // CHECK: @llvm.x86.3dnow.pfrcpit2
  return _m_pfrcpit2(m1, m2);
}

__m64 test_m_pfrsqrt(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pfrsqrt
  // GCC-LABEL: define double @test_m_pfrsqrt
  // CHECK: @llvm.x86.3dnow.pfrsqrt
  return _m_pfrsqrt(m);
}

__m64 test_m_pfrsqrtit1(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfrsqrtit1
  // GCC-LABEL: define double @test_m_pfrsqrtit1
  // CHECK: @llvm.x86.3dnow.pfrsqit1
  return _m_pfrsqrtit1(m1, m2);
}

__m64 test_m_pfsub(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfsub
  // GCC-LABEL: define double @test_m_pfsub
  // CHECK: @llvm.x86.3dnow.pfsub
  return _m_pfsub(m1, m2);
}

__m64 test_m_pfsubr(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfsubr
  // GCC-LABEL: define double @test_m_pfsubr
  // CHECK: @llvm.x86.3dnow.pfsubr
  return _m_pfsubr(m1, m2);
}

__m64 test_m_pi2fd(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pi2fd
  // GCC-LABEL: define double @test_m_pi2fd
  // CHECK: @llvm.x86.3dnow.pi2fd
  return _m_pi2fd(m);
}

__m64 test_m_pmulhrw(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pmulhrw
  // GCC-LABEL: define double @test_m_pmulhrw
  // CHECK: @llvm.x86.3dnow.pmulhrw
  return _m_pmulhrw(m1, m2);
}

__m64 test_m_pf2iw(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pf2iw
  // GCC-LABEL: define double @test_m_pf2iw
  // CHECK: @llvm.x86.3dnowa.pf2iw
  return _m_pf2iw(m);
}

__m64 test_m_pfnacc(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfnacc
  // GCC-LABEL: define double @test_m_pfnacc
  // CHECK: @llvm.x86.3dnowa.pfnacc
  return _m_pfnacc(m1, m2);
}

__m64 test_m_pfpnacc(__m64 m1, __m64 m2) {
  // PS4-LABEL: define i64 @test_m_pfpnacc
  // GCC-LABEL: define double @test_m_pfpnacc
  // CHECK: @llvm.x86.3dnowa.pfpnacc
  return _m_pfpnacc(m1, m2);
}

__m64 test_m_pi2fw(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pi2fw
  // GCC-LABEL: define double @test_m_pi2fw
  // CHECK: @llvm.x86.3dnowa.pi2fw
  return _m_pi2fw(m);
}

__m64 test_m_pswapdsf(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pswapdsf
  // GCC-LABEL: define double @test_m_pswapdsf
  // CHECK: @llvm.x86.3dnowa.pswapd
  return _m_pswapdsf(m);
}

__m64 test_m_pswapdsi(__m64 m) {
  // PS4-LABEL: define i64 @test_m_pswapdsi
  // GCC-LABEL: define double @test_m_pswapdsi
  // CHECK: @llvm.x86.3dnowa.pswapd
  return _m_pswapdsi(m);
}
