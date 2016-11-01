// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -faltivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefix=CHECK-BE

// RUN: %clang_cc1 -faltivec -target-feature +power9-vector \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <altivec.h>

vector signed char vsca, vscb;
vector unsigned char vuca, vucb;
vector bool char vbca, vbcb;
vector signed short vssa, vssb;
vector unsigned short vusa, vusb;
vector bool short vbsa, vbsb;
vector signed int vsia, vsib;
vector unsigned int vuia, vuib;
vector bool int vbia, vbib;
vector signed long long vsla, vslb;
vector unsigned long long vula, vulb;
vector bool long long vbla, vblb;
vector float vfa, vfb;
vector double vda, vdb;
vector unsigned __int128 vui128a, vui128b;
vector signed __int128 vsi128a, vsi128b;

unsigned test1(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_match_index (vsca, vscb);
}
unsigned test2(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_match_index (vuca, vucb);
}
unsigned test3(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_match_index (vsia, vsib);
}
unsigned test4(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_match_index (vuia, vuib);
}
unsigned test5(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_match_index (vssa, vssb);
}
unsigned test6(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_match_index (vusa, vusb);
}
unsigned test7(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: or <16 x i8>
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: or <16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: or <16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: or <16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_match_or_eos_index (vsca, vscb);
}
unsigned test8(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: or <16 x i8>
// CHECK-BE: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK-BE: or <16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: or <16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpequb(<16 x i8>
// CHECK: or <16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_match_or_eos_index (vuca, vucb);
}
unsigned test9(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: or <4 x i32>
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: or <4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: or <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: or <4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_match_or_eos_index (vsia, vsib);
}
unsigned test10(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: or <4 x i32>
// CHECK-BE: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK-BE: or <4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: or <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpequw(<4 x i32>
// CHECK: or <4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_match_or_eos_index (vuia, vuib);
}
unsigned test11(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: or <8 x i16>
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: or <8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: or <8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: or <8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_match_or_eos_index (vssa, vssb);
}
unsigned test12(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: or <8 x i16>
// CHECK-BE: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK-BE: or <8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: or <8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpequh(<8 x i16>
// CHECK: or <8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_match_or_eos_index (vusa, vusb);
}
unsigned test13(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_mismatch_index (vsca, vscb);
}
unsigned test14(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_mismatch_index (vuca, vucb);
}
unsigned test15(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_mismatch_index (vsia, vsib);
}
unsigned test16(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_mismatch_index (vuia, vuib);
}
unsigned test17(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_mismatch_index (vssa, vssb);
}
unsigned test18(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_mismatch_index (vusa, vusb);
}
unsigned test19(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpnezb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_mismatch_or_eos_index (vsca, vscb);
}
unsigned test20(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezb(<16 x i8>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 3
// CHECK: @llvm.ppc.altivec.vcmpnezb(<16 x i8>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 3
  return vec_first_mismatch_or_eos_index (vuca, vucb);
}
unsigned test21(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezw(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpnezw(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_mismatch_or_eos_index (vsia, vsib);
}
unsigned test22(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezw(<4 x i32>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 5
// CHECK: @llvm.ppc.altivec.vcmpnezw(<4 x i32>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 5
  return vec_first_mismatch_or_eos_index (vuia, vuib);
}
unsigned test23(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpnezh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_mismatch_or_eos_index (vssa, vssb);
}
unsigned test24(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnezh(<8 x i16>
// CHECK-BE: @llvm.ctlz.v2i64(<2 x i64>
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: icmp eq i64 {{.*}}, 64
// CHECK-BE: extractelement <2 x i64>
// CHECK-BE: add i64 {{.*}}, 64
// CHECK-BE: lshr i64 {{.*}}, 4
// CHECK: @llvm.ppc.altivec.vcmpnezh(<8 x i16>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK: extractelement <2 x i64>
// CHECK: icmp eq i64 {{.*}}, 64
// CHECK: extractelement <2 x i64>
// CHECK: add i64 {{.*}}, 64
// CHECK: lshr i64 {{.*}}, 4
  return vec_first_mismatch_or_eos_index (vusa, vusb);
}
vector bool char test25(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_cmpne (vbca, vbcb);
}
vector bool char test26(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_cmpne (vsca, vscb);
}
vector bool char test27(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.altivec.vcmpneb(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_cmpne (vuca, vucb);
}
vector bool int test28(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cmpne (vbia, vbib);
}
vector bool int test29(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cmpne (vsia, vsib);
}
vector bool int test30(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cmpne (vuia, vuib);
}
vector bool long long test31(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK-BE: xor <2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK: xor <2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cmpne (vbla, vblb);
}
vector bool long long test32(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK-BE: xor <2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK: xor <2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cmpne (vsla, vslb);
}
vector bool long long test33(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK-BE: xor <2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK: xor <2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cmpne (vula, vulb);
}
vector bool short test34(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_cmpne (vbsa, vbsb);
}
vector bool short test35(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_cmpne (vssa, vssb);
}
vector bool short test36(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.ppc.altivec.vcmpneh(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_cmpne (vusa, vusb);
}
vector bool long long test37(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK-BE: xor <2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vcmpequd(<2 x i64>
// CHECK: xor <2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cmpne (vda, vdb);
}
vector bool int test38(void) {
// CHECK-BE: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vcmpnew(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cmpne (vfa, vfb);
}
vector signed char test39(void) {
// CHECK-BE: @llvm.cttz.v16i8(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.cttz.v16i8(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_cnttz (vsca);
}
vector unsigned char test40(void) {
// CHECK-BE: @llvm.cttz.v16i8(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.cttz.v16i8(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_cnttz (vuca);
}
vector signed int test41(void) {
// CHECK-BE: @llvm.cttz.v4i32(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.cttz.v4i32(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cnttz (vsia);
}
vector unsigned int test42(void) {
// CHECK-BE: @llvm.cttz.v4i32(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.cttz.v4i32(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_cnttz (vuia);
}
vector signed long long test43(void) {
// CHECK-BE: @llvm.cttz.v2i64(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cnttz (vsla);
}
vector unsigned long long test44(void) {
// CHECK-BE: @llvm.cttz.v2i64(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.cttz.v2i64(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_cnttz (vula);
}
vector signed short test45(void) {
// CHECK-BE: @llvm.cttz.v8i16(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.cttz.v8i16(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_cnttz (vssa);
}
vector unsigned short test46(void) {
// CHECK-BE: @llvm.cttz.v8i16(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.cttz.v8i16(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_cnttz (vusa);
}
vector unsigned char test47(void) {
// CHECK-BE: @llvm.ctpop.v16i8(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ctpop.v16i8(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_popcnt (vsca);
}
vector unsigned char test48(void) {
// CHECK-BE: @llvm.ctpop.v16i8(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ctpop.v16i8(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_popcnt (vuca);
}
vector unsigned int test49(void) {
// CHECK-BE: @llvm.ctpop.v4i32(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ctpop.v4i32(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_popcnt (vsia);
}
vector unsigned int test50(void) {
// CHECK-BE: @llvm.ctpop.v4i32(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ctpop.v4i32(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_popcnt (vuia);
}
vector unsigned long long test51(void) {
// CHECK-BE: @llvm.ctpop.v2i64(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ctpop.v2i64(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_popcnt (vsla);
}
vector unsigned long long test52(void) {
// CHECK-BE: @llvm.ctpop.v2i64(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ctpop.v2i64(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_popcnt (vula);
}
vector unsigned short test53(void) {
// CHECK-BE: @llvm.ctpop.v8i16(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.ctpop.v8i16(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_popcnt (vssa);
}
vector unsigned short test54(void) {
// CHECK-BE: @llvm.ctpop.v8i16(<8 x i16>
// CHECK-BE-NEXT: ret <8 x i16>
// CHECK: @llvm.ctpop.v8i16(<8 x i16>
// CHECK-NEXT: ret <8 x i16>
  return vec_popcnt (vusa);
}
vector double test55(void) {
// CHECK-BE: @llvm.ppc.vsx.xviexpdp(<2 x i64> %{{.+}}, <2 x i64>
// CHECK-BE-NEXT: ret <2 x double>
// CHECK: @llvm.ppc.vsx.xviexpdp(<2 x i64> %{{.+}}, <2 x i64>
// CHECK-NEXT: ret <2 x double>
  return vec_insert_exp (vda,vulb);
}
vector double test56(void) {
// CHECK-BE: @llvm.ppc.vsx.xviexpdp(<2 x i64> %{{.+}}, <2 x i64>
// CHECK-BE-NEXT: ret <2 x double>
// CHECK: @llvm.ppc.vsx.xviexpdp(<2 x i64> %{{.+}}, <2 x i64>
// CHECK-NEXT: ret <2 x double>
  return vec_insert_exp (vula, vulb);
}
vector float test57(void) {
// CHECK-BE: @llvm.ppc.vsx.xviexpsp(<4 x i32> %{{.+}}, <4 x i32>
// CHECK-BE-NEXT: ret <4 x float>
// CHECK: @llvm.ppc.vsx.xviexpsp(<4 x i32> %{{.+}}, <4 x i32>
// CHECK-NEXT: ret <4 x float>
  return vec_insert_exp (vfa,vuib);
}
vector float test58(void) {
// CHECK-BE: @llvm.ppc.vsx.xviexpsp(<4 x i32> %{{.+}}, <4 x i32>
// CHECK-BE-NEXT: ret <4 x float>
// CHECK: @llvm.ppc.vsx.xviexpsp(<4 x i32> %{{.+}}, <4 x i32>
// CHECK-NEXT: ret <4 x float>
  return vec_insert_exp (vuia,vuib);
}
signed int test59(void) {
// CHECK-BE: @llvm.ppc.altivec.vclzlsbb(<16 x i8>
// CHECK-BE-NEXT: ret i32
// CHECK-LE: @llvm.ppc.altivec.vctzlsbb(<16 x i8>
// CHECK-LE-NEXT: ret i32
  return vec_cntlz_lsbb (vuca);
}
signed int test60(void) {
// CHECK-BE: @llvm.ppc.altivec.vclzlsbb(<16 x i8>
// CHECK-BE-NEXT: ret i32
// CHECK-LE: @llvm.ppc.altivec.vctzlsbb(<16 x i8>
// CHECK-LE-NEXT: ret i32
  return vec_cntlz_lsbb (vsca);
}
signed int test61(void) {
// CHECK-BE: @llvm.ppc.altivec.vctzlsbb(<16 x i8>
// CHECK-BE-NEXT: ret i32
// CHECK-LE: @llvm.ppc.altivec.vclzlsbb(<16 x i8>
// CHECK-LE-NEXT: ret i32
  return vec_cnttz_lsbb (vsca);
}
signed int test62(void) {
// CHECK-BE: @llvm.ppc.altivec.vctzlsbb(<16 x i8>
// CHECK-BE-NEXT: ret i32
// CHECK-LE: @llvm.ppc.altivec.vclzlsbb(<16 x i8>
// CHECK-LE-NEXT: ret i32
  return vec_cnttz_lsbb (vuca);
}

vector unsigned int test63(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybw(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vprtybw(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_parity_lsbb (vuia);
}

vector unsigned int test64(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybw(<4 x i32>
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vprtybw(<4 x i32>
// CHECK-NEXT: ret <4 x i32>
  return vec_parity_lsbb (vsia);
}

vector unsigned long long test65(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybd(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vprtybd(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_parity_lsbb (vula);
}

vector unsigned long long test66(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybd(<2 x i64>
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vprtybd(<2 x i64>
// CHECK-NEXT: ret <2 x i64>
  return vec_parity_lsbb (vsla);
}
vector unsigned __int128 test67(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybq(<1 x i128>
// CHECK-BE-NEXT: ret <1 x i128>
// CHECK: @llvm.ppc.altivec.vprtybq(<1 x i128>
// CHECK-NEXT: ret <1 x i128>
  return vec_parity_lsbb (vui128a);
}

vector unsigned __int128 test68(void) {
// CHECK-BE: @llvm.ppc.altivec.vprtybq(<1 x i128>
// CHECK-BE-NEXT: ret <1 x i128>
// CHECK: @llvm.ppc.altivec.vprtybq(<1 x i128>
// CHECK-NEXT: ret <1 x i128>
  return vec_parity_lsbb (vsi128a);
}

vector unsigned char test69(void) {
// CHECK-BE: call <16 x i8> @llvm.ppc.altivec.vabsdub(<16 x i8> {{.+}}, <16 x i8> {{.+}})
// CHECK: call <16 x i8> @llvm.ppc.altivec.vabsdub(<16 x i8> {{.+}}, <16 x i8> {{.+}})
  return vec_absd(vuca, vucb);
}
vector unsigned short test70(void) {
// CHECK-BE: call <8 x i16> @llvm.ppc.altivec.vabsduh(<8 x i16> {{.+}}, <8 x i16> {{.+}})
// CHECK: call <8 x i16> @llvm.ppc.altivec.vabsduh(<8 x i16> {{.+}}, <8 x i16> {{.+}})
  return vec_absd(vusa, vusb);
}
vector unsigned int test71(void) {
// CHECK-BE: call <4 x i32> @llvm.ppc.altivec.vabsduw(<4 x i32> {{.+}}, <4 x i32> {{.+}})
// CHECK: call <4 x i32> @llvm.ppc.altivec.vabsduw(<4 x i32> {{.+}}, <4 x i32> {{.+}})
  return vec_absd(vuia, vuib);
}
