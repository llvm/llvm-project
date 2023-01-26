// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +altivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -flax-vector-conversions=integer \
// RUN:   -o - | FileCheck %s -check-prefix=CHECK-BE

// RUN: %clang_cc1 -target-feature +altivec -target-feature +power9-vector \
// RUN:   -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -flax-vector-conversions=integer \
// RUN:   -o - | FileCheck %s

// RUN: %clang_cc1 -flax-vector-conversions=none -target-feature +altivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefix=CHECK-BE

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

float f[4] = { 23.4f, 56.7f, 89.0f, 12.3f };
double d[2] = { 23.4, 56.7 };
signed char sc[16] = { -8,  9, -10, 11, -12, 13, -14, 15,
                        -0,  1,  -2,  3,  -4,  5,  -6,  7 };
unsigned char uc[16] = { 8,  9, 10, 11, 12, 13, 14, 15,
                          0,  1,  2,  3,  4,  5,  6,  7 };
signed short ss[8] = { -1, 2, -3, 4, -5, 6, -7, 8 };
unsigned short us[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
signed int si[4] = { -1, 2, -3, 4 };
unsigned int ui[4] = { 0, 1, 2, 3 };
signed long sl[2] = { -1L, 2L };
unsigned long ul[2] = { 1L, 2L };
signed long long sll[2] = { 1LL, 1LL };
unsigned long long ull[2] = { -1LL, 1LL };
signed __int128 sint128[1] = { -1 };
unsigned __int128 uint128[1] = { 1 };

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
vector unsigned char test72(void) {
// CHECK-BE: @llvm.ppc.altivec.vslv(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.altivec.vslv(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_slv (vuca, vucb);
}
vector unsigned char test73(void) {
// CHECK-BE: @llvm.ppc.altivec.vsrv(<16 x i8>
// CHECK-BE-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.altivec.vsrv(<16 x i8>
// CHECK-NEXT: ret <16 x i8>
  return vec_srv (vuca, vucb);
}
vector unsigned short test74(void) {
// CHECK-BE: @llvm.ppc.vsx.xvcvsphp(<4 x float>
// CHECK-BE: @llvm.ppc.vsx.xvcvsphp(<4 x float>
// CHECK-BE: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.xvcvsphp(<4 x float>
// CHECK: @llvm.ppc.vsx.xvcvsphp(<4 x float>
// CHECK: @llvm.ppc.altivec.vperm
  return vec_pack_to_short_fp32(vfa, vfb);
}
vector unsigned int test75(void) {
// CHECK-BE: @llvm.ppc.altivec.vrlwmi(<4 x i32
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.altivec.vrlwmi(<4 x i32
// CHECK-NEXT: ret <4 x i32>
  return vec_rlmi(vuia, vuia, vuia);
}
vector unsigned long long test76(void) {
// CHECK-BE: @llvm.ppc.altivec.vrldmi(<2 x i64
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.altivec.vrldmi(<2 x i64
// CHECK-NEXT: ret <2 x i64>
  return vec_rlmi(vula, vula, vula);
}
vector unsigned int test77(void) {
// CHECK-BE: %[[RES1:.+]] = shl <4 x i32
// CHECK-BE: %[[RES2:.+]] = or <4 x i32> %[[RES1]]
// CHECK-BE: @llvm.ppc.altivec.vrlwnm(<4 x i32
// CHECK-BE: ret <4 x i32>
// CHECK: %[[RES1:.+]] = shl <4 x i32
// CHECK: %[[RES2:.+]] = or <4 x i32> %[[RES1]]
// CHECK: @llvm.ppc.altivec.vrlwnm(<4 x i32
// CHECK: ret <4 x i32>
  return vec_rlnm(vuia, vuia, vuia);
}
vector unsigned long long test78(void) {
// CHECK-BE: %[[RES1:.+]] = shl <2 x i64
// CHECK-BE: %[[RES2:.+]] = or <2 x i64> %[[RES1]]
// CHECK-BE: @llvm.ppc.altivec.vrldnm(<2 x i64
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: %[[RES1:.+]] = shl <2 x i64
// CHECK: %[[RES2:.+]] = or <2 x i64> %[[RES1]]
// CHECK: @llvm.ppc.altivec.vrldnm(<2 x i64
// CHECK-NEXT: ret <2 x i64>
  return vec_rlnm(vula, vula, vula);
}
vector double test79(void) {
// CHECK-BE: extractelement <4 x float>
// CHECK-BE: fpext float
// CHECK-BE: insertelement <2 x double>
// CHECK-BE: extractelement <4 x float>
// CHECK-BE: fpext float
// CHECK-BE: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
  return vec_unpackh(vfa);
}
vector double test80(void) {
// CHECK-BE: extractelement <4 x float>
// CHECK-BE: fpext float
// CHECK-BE: insertelement <2 x double>
// CHECK-BE: extractelement <4 x float>
// CHECK-BE: fpext float
// CHECK-BE: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
// CHECK: extractelement <4 x float>
// CHECK: fpext float
// CHECK: insertelement <2 x double>
  return vec_unpackl(vfa);
}
vector float test81(void) {
  // CHECK: extractelement <2 x double>
  // CHECK: fptrunc double
  // CHECK: insertelement <4 x float>
  // CHECK: extractelement <2 x double>
  // CHECK: fptrunc double
  // CHECK: insertelement <4 x float>
  // CHECK: extractelement <2 x double>
  // CHECK: fptrunc double
  // CHECK: insertelement <4 x float>
  // CHECK: extractelement <2 x double>
  // CHECK: fptrunc double
  // CHECK: insertelement <4 x float>
  // CHECK-LE: extractelement <2 x double>
  // CHECK-LE: fptrunc double
  // CHECK-LE: insertelement <4 x float>
  // CHECK-LE: extractelement <2 x double>
  // CHECK-LE: fptrunc double
  // CHECK-LE: insertelement <4 x float>
  // CHECK-LE: extractelement <2 x double>
  // CHECK-LE: fptrunc double
  // CHECK-LE: insertelement <4 x float>
  // CHECK-LE: extractelement <2 x double>
  // CHECK-LE: fptrunc double
  // CHECK-LE: insertelement <4 x float>
  return vec_pack(vda, vdb);
}
vector unsigned int test82(void) {
// CHECK-BE: @llvm.ppc.vsx.xvxexpsp(<4 x float> {{.+}})
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.xvxexpsp(<4 x float> {{.+}})
// CHECK-NEXT: ret <4 x i32>
  return vec_extract_exp(vfa);
}
vector unsigned long long test83(void) {
// CHECK-BE: @llvm.ppc.vsx.xvxexpdp(<2 x double> {{.+}})
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.xvxexpdp(<2 x double> {{.+}})
// CHECK-NEXT: ret <2 x i64>
  return vec_extract_exp(vda);
}
vector unsigned int test84(void) {
// CHECK-BE: @llvm.ppc.vsx.xvxsigsp(<4 x float> {{.+}})
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.xvxsigsp(<4 x float> {{.+}})
// CHECK-NEXT: ret <4 x i32>
  return vec_extract_sig(vfa);
}
vector unsigned long long test85(void) {
// CHECK-BE: @llvm.ppc.vsx.xvxsigdp(<2 x double> {{.+}})
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.xvxsigdp(<2 x double> {{.+}})
// CHECK-NEXT: ret <2 x i64>
  return vec_extract_sig(vda);
}
vector bool int test86(void) {
// CHECK-BE: @llvm.ppc.vsx.xvtstdcsp(<4 x float> {{.+}}, i32 127)
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.xvtstdcsp(<4 x float> {{.+}}, i32 127)
// CHECK-NEXT: ret <4 x i32>
   return vec_test_data_class(vfa, __VEC_CLASS_FP_NOT_NORMAL);
}
vector bool long long test87(void) {
// CHECK-BE: @llvm.ppc.vsx.xvtstdcdp(<2 x double> {{.+}}, i32 127)
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.xvtstdcdp(<2 x double> {{.+}}, i32 127)
// CHECK-NEXT: ret <2 x i64>
  return vec_test_data_class(vda, __VEC_CLASS_FP_NOT_NORMAL);
}
vector unsigned char test88(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <16 x i8>
  return vec_xl_len(uc,0);
}
vector signed char test89(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <16 x i8>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <16 x i8>
  return vec_xl_len(sc,0);
}
vector unsigned short test90(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <8 x i16>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <8 x i16>
  return vec_xl_len(us,0);
}
vector signed short test91(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <8 x i16>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <8 x i16>
  return vec_xl_len(ss,0);
}
vector unsigned int test92(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT: ret <4 x i32>
  return vec_xl_len(ui,0);
}

vector signed int test93(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT: ret <4 x i32>
  return vec_xl_len(si,0);
}

vector float test94(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <4 x i32>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <4 x i32>
  return vec_xl_len(f,0);
}

vector unsigned long long test95(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <2 x i64>
  return vec_xl_len(ull,0);
}
 
vector signed long long test96(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <2 x i64>
  return vec_xl_len(sll,0);
}

vector double test97(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <2 x i64>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <2 x i64>
  return vec_xl_len(d,0);
}

vector unsigned __int128 test98(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <1 x i128>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <1 x i128>
  return vec_xl_len(uint128,0);
}

vector signed __int128 test99(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-BE-NEXT-NEXT: ret <1 x i128>
// CHECK: @llvm.ppc.vsx.lxvl(ptr %{{.+}}, i64
// CHECK-NEXT-NEXT: ret <1 x i128>
  return vec_xl_len(sint128,0);
}

void test100(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
  return vec_xst_len(vuca,uc,0);
}

void test101(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
  return vec_xst_len(vsca,sc,0);
}

void test102(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vusa,us,0);
}

void test103(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vssa,ss,0);
}

void test104(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vuia,ui,0);
}

void test105(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vsia,si,0);
}

void test106(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vfa,f,0);
}

void test107(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vula,ull,0);
}

void test108(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vsla,sll,0);
}

void test109(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vda,d,0);
}

void test110(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vui128a,uint128,0);
}

void test111(void) {
// CHECK-BE: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.vsx.stxvl(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
 return vec_xst_len(vsi128a,sint128,0);
}

vector unsigned char test112(void) {
// CHECK-BE: @llvm.ppc.vsx.lxvll(ptr %{{.+}}, i64
// CHECK-BE: @llvm.ppc.altivec.lvsr(ptr %{{.+}}
// CHECK-BE: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.lxvll(ptr %{{.+}}, i64
// CHECK: @llvm.ppc.altivec.lvsr(ptr %{{.+}}
// CHECK: @llvm.ppc.altivec.vperm
  return vec_xl_len_r(uc,0);
}
void test113(void) {
// CHECK-BE: @llvm.ppc.altivec.lvsl(ptr %{{.+}}
// CHECK-BE: @llvm.ppc.altivec.vperm
// CHECK-BE: @llvm.ppc.vsx.stxvll(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
// CHECK: @llvm.ppc.altivec.lvsl(ptr %{{.+}}
// CHECK: @llvm.ppc.altivec.vperm
// CHECK: @llvm.ppc.vsx.stxvll(<4 x i32> %{{.+}}, ptr %{{.+}}, i64
  return vec_xst_len_r(vuca,uc,0);
}
vector float test114(void) {
// CHECK-BE: shufflevector <8 x i16> {{.+}}, <8 x i16> {{.+}}, <8 x i32> <i32 undef, i32 0, i32 undef, i32 1, i32 undef, i32 2, i32 undef, i32 3>
// CHECK-BE: @llvm.ppc.vsx.xvcvhpsp(<8 x i16> {{.+}})
// CHECK-BE-NEXT: ret <4 x float>
// CHECK: shufflevector <8 x i16> {{.+}}, <8 x i16> {{.+}}, <8 x i32> <i32 0, i32 undef, i32 1, i32 undef, i32 2, i32 undef, i32 3, i32 undef>
// CHECK: @llvm.ppc.vsx.xvcvhpsp(<8 x i16> {{.+}})
// CHECK-NEXT: ret <4 x float>
  return vec_extract_fp32_from_shorth(vusa);
}
vector float test115(void) {
// CHECK-BE: shufflevector <8 x i16> {{.+}}, <8 x i16> {{.+}}, <8 x i32> <i32 undef, i32 4, i32 undef, i32 5, i32 undef, i32 6, i32 undef, i32 7>
// CHECK-BE: @llvm.ppc.vsx.xvcvhpsp(<8 x i16> {{.+}})
// CHECK-BE-NEXT: ret <4 x float>
// CHECK: shufflevector <8 x i16> {{.+}}, <8 x i16> {{.+}}, <8 x i32> <i32 4, i32 undef, i32 5, i32 undef, i32 6, i32 undef, i32 7, i32 undef>
// CHECK: @llvm.ppc.vsx.xvcvhpsp(<8 x i16> {{.+}})
// CHECK-NEXT: ret <4 x float>
  return vec_extract_fp32_from_shortl(vusa);
}
vector unsigned char test116(void) {
// CHECK-BE: [[T1:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32> {{.+}}, <2 x i64> {{.+}}, i32 7)
// CHECK-BE-NEXT: bitcast <4 x i32> [[T1]] to <16 x i8>
// CHECK: [[T1:%.+]] = shufflevector <2 x i64> {{.+}}, <2 x i64> {{.+}}, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT: [[T2:%.+]] =  bitcast <2 x i64> [[T1]] to <4 x i32>
// CHECK-NEXT: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32> [[T2]], <2 x i64> {{.+}}, i32 5)
  return vec_insert4b(vuia, vuca, 7);
}
vector unsigned char test117(void) {
// CHECK-BE: [[T1:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32> {{.+}}, <2 x i64> {{.+}}, i32 12)
// CHECK-BE-NEXT: bitcast <4 x i32> [[T1]] to <16 x i8>
// CHECK: [[T1:%.+]] = shufflevector <2 x i64> {{.+}}, <2 x i64> {{.+}}, <2 x i32> <i32 1, i32 0>
// CHECK-NEXT: [[T2:%.+]] =  bitcast <2 x i64> [[T1]] to <4 x i32>
// CHECK-NEXT: [[T3:%.+]] = call <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32> [[T2]], <2 x i64> {{.+}}, i32 0)
  return vec_insert4b(vuia, vuca, 13);
}
vector unsigned long long test118(void) {
// CHECK-BE: call <2 x i64> @llvm.ppc.vsx.xxextractuw(<2 x i64> {{.+}}, i32 11)
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: [[T1:%.+]] = call <2 x i64> @llvm.ppc.vsx.xxextractuw(<2 x i64> {{.+}}, i32 1)
// CHECK-NEXT: shufflevector <2 x i64> [[T1]], <2 x i64> [[T1]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT: ret <2 x i64>
  return vec_extract4b(vuca, 11);
}
vector unsigned long long test119(void) {
// CHECK-BE: call <2 x i64> @llvm.ppc.vsx.xxextractuw(<2 x i64> {{.+}}, i32 0)
// CHECK-BE-NEXT: ret <2 x i64>
// CHECK: [[T1:%.+]] = call <2 x i64> @llvm.ppc.vsx.xxextractuw(<2 x i64> {{.+}}, i32 12)
// CHECK-NEXT: shufflevector <2 x i64> [[T1]], <2 x i64> [[T1]], <2 x i32> <i32 1, i32 0>
// CHECK-NEXT: ret <2 x i64>
  return vec_extract4b(vuca, -5);
}

vector signed int test_vec_signexti_si_sc(void) {
    // CHECK: @llvm.ppc.altivec.vextsb2w(<16 x i8>
    // CHECK-NEXT: ret <4 x i32>
    return vec_signexti(vsca);
}

vector signed int test_vec_signexti_si_ss(void) {
    // CHECK: @llvm.ppc.altivec.vextsh2w(<8 x i16>
    // CHECK-NEXT: ret <4 x i32>
    return vec_signexti(vssa);
}

vector signed long long test_vec_signextll_sll_sc(void) {
    // CHECK: @llvm.ppc.altivec.vextsb2d(<16 x i8>
    // CHECK-NEXT: ret <2 x i64>
    return vec_signextll(vsca);
}

vector signed long long test_vec_signextll_sll_ss(void) {
    // CHECK: @llvm.ppc.altivec.vextsh2d(<8 x i16>
    // CHECK-NEXT: ret <2 x i64>
    return vec_signextll(vssa);
}

vector signed long long test_vec_signextll_sll_si(void) {
    // CHECK: @llvm.ppc.altivec.vextsw2d(<4 x i32>
    // CHECK-NEXT: ret <2 x i64>
    return vec_signextll(vsia);
}

vector unsigned long long test_vbpermd(void) {
  // CHECK: @llvm.ppc.altivec.vbpermd(<2 x i64>
  // CHECK-BE: @llvm.ppc.altivec.vbpermd(<2 x i64>
  return vec_bperm(vula, vuca);
}
