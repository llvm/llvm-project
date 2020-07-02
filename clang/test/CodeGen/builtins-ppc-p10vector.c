// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +vsx -target-feature +altivec \
// RUN:   -target-cpu pwr10 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s

#include <altivec.h>

vector signed char vsca, vscb;
vector unsigned char vuca, vucb, vucc;
vector signed short vssa, vssb;
vector unsigned short vusa, vusb, vusc;
vector signed int vsia, vsib;
vector unsigned int vuia, vuib, vuic;
vector signed long long vslla, vsllb;
vector unsigned long long vulla, vullb, vullc;
vector unsigned __int128 vui128a, vui128b, vui128c;
vector float vfa, vfb;
vector double vda, vdb;
unsigned int uia;

vector unsigned long long test_vpdepd(void) {
  // CHECK: @llvm.ppc.altivec.vpdepd(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_pdep(vulla, vullb);
}

vector unsigned long long test_vpextd(void) {
  // CHECK: @llvm.ppc.altivec.vpextd(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_pext(vulla, vullb);
}

vector unsigned long long test_vcfuged(void) {
  // CHECK: @llvm.ppc.altivec.vcfuged(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_cfuge(vulla, vullb);
}

unsigned long long test_vgnb_1(void) {
  // CHECK: @llvm.ppc.altivec.vgnb(<1 x i128> %{{.+}}, i32 2)
  // CHECK-NEXT: ret i64
  return vec_gnb(vui128a, 2);
}

unsigned long long test_vgnb_2(void) {
  // CHECK: @llvm.ppc.altivec.vgnb(<1 x i128> %{{.+}}, i32 7)
  // CHECK-NEXT: ret i64
  return vec_gnb(vui128a, 7);
}

unsigned long long test_vgnb_3(void) {
  // CHECK: @llvm.ppc.altivec.vgnb(<1 x i128> %{{.+}}, i32 5)
  // CHECK-NEXT: ret i64
  return vec_gnb(vui128a, 5);
}

vector unsigned char test_xxeval_uc(void) {
  // CHECK: @llvm.ppc.vsx.xxeval(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32 0)
  // CHECK: ret <16 x i8>
  return vec_ternarylogic(vuca, vucb, vucc, 0);
}

vector unsigned short test_xxeval_us(void) {
  // CHECK: @llvm.ppc.vsx.xxeval(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32 255)
  // CHECK: ret <8 x i16>
  return vec_ternarylogic(vusa, vusb, vusc, 255);
}

vector unsigned int test_xxeval_ui(void) {
  // CHECK: @llvm.ppc.vsx.xxeval(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32 150)
  // CHECK: ret <4 x i32>
  return vec_ternarylogic(vuia, vuib, vuic, 150);
}

vector unsigned long long test_xxeval_ull(void) {
  // CHECK: @llvm.ppc.vsx.xxeval(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32 1)
  // CHECK: ret <2 x i64>
  return vec_ternarylogic(vulla, vullb, vullc, 1);
}

vector unsigned __int128 test_xxeval_ui128(void) {
  // CHECK: @llvm.ppc.vsx.xxeval(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32 246)
  // CHECK: ret <1 x i128>
  return vec_ternarylogic(vui128a, vui128b, vui128c, 246);
}

vector unsigned char test_xxgenpcvbm(void) {
  // CHECK: @llvm.ppc.vsx.xxgenpcvbm(<16 x i8> %{{.+}}, i32
  // CHECK-NEXT: ret <16 x i8>
  return vec_genpcvm(vuca, 0);
}

vector unsigned short test_xxgenpcvhm(void) {
  // CHECK: @llvm.ppc.vsx.xxgenpcvhm(<8 x i16> %{{.+}}, i32
  // CHECK-NEXT: ret <8 x i16>
  return vec_genpcvm(vusa, 0);
}

vector unsigned int test_xxgenpcvwm(void) {
  // CHECK: @llvm.ppc.vsx.xxgenpcvwm(<4 x i32> %{{.+}}, i32
  // CHECK-NEXT: ret <4 x i32>
  return vec_genpcvm(vuia, 0);
}

vector unsigned long long test_xxgenpcvdm(void) {
  // CHECK: @llvm.ppc.vsx.xxgenpcvdm(<2 x i64> %{{.+}}, i32
  // CHECK-NEXT: ret <2 x i64>
  return vec_genpcvm(vulla, 0);
}

vector signed char test_vec_vclrl_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrl(vsca, uia);
}

vector unsigned char test_vec_clrl_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrl(vuca, uia);
}

vector signed char test_vec_vclrr_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrr(vsca, uia);
}

vector unsigned char test_vec_clrr_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vclrrb(<16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vclrlb(<16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_clrr(vuca, uia);
}

vector unsigned long long test_vclzdm(void) {
  // CHECK: @llvm.ppc.altivec.vclzdm(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_cntlzm(vulla, vullb);
}

vector unsigned long long test_vctzdm(void) {
  // CHECK: @llvm.ppc.altivec.vctzdm(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_cnttzm(vulla, vullb);
}

vector signed char test_vec_sldb_sc(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 0
  // CHECK-NEXT: ret <16 x i8>
  return vec_sldb(vsca, vscb, 0);
  }

vector unsigned char test_vec_sldb_uc(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 1
  // CHECK-NEXT: ret <16 x i8>
  return vec_sldb(vuca, vucb, 1);
}

vector signed short test_vec_sldb_ss(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 2
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_sldb(vssa, vssb, 2);
}

vector unsigned short test_vec_sldb_us(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 3
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_sldb(vusa, vusb, 3);
}

vector signed int test_vec_sldb_si(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 4
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_sldb(vsia, vsib, 4);
}

vector unsigned int test_vec_sldb_ui(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 5
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_sldb(vuia, vuib, 5);
}

vector signed long long test_vec_sldb_sll(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 6
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_sldb(vslla, vsllb, 6);
}

vector unsigned long long test_vec_sldb_ull(void) {
  // CHECK: @llvm.ppc.altivec.vsldbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 7
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_sldb(vulla, vullb, 7);
}

vector signed char test_vec_srdb_sc(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 0
  // CHECK-NEXT: ret <16 x i8>
  return vec_srdb(vsca, vscb, 8);
}

vector unsigned char test_vec_srdb_uc(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 1
  // CHECK-NEXT: ret <16 x i8>
  return vec_srdb(vuca, vucb, 9);
}

vector signed short test_vec_srdb_ss(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 2
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_srdb(vssa, vssb, 10);
}

vector unsigned short test_vec_srdb_us(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 3
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_srdb(vusa, vusb, 3);
}

vector signed int test_vec_srdb_si(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 4
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_srdb(vsia, vsib, 4);
}

vector unsigned int test_vec_srdb_ui(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 5
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_srdb(vuia, vuib, 5);
}

vector signed long long test_vec_srdb_sll(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 6
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_srdb(vslla, vsllb, 6);
}

vector unsigned long long test_vec_srdb_ull(void) {
  // CHECK: @llvm.ppc.altivec.vsrdbi(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32 7
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_srdb(vulla, vullb, 7);
}

vector signed char test_vec_permx_sc(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: ret <16 x i8>
  return vec_permx(vsca, vscb, vucc, 0);
}

vector unsigned char test_vec_permx_uc(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: ret <16 x i8>
  return vec_permx(vuca, vucb, vucc, 1);
}

vector signed short test_vec_permx_ss(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_permx(vssa, vssb, vucc, 2);
}

vector unsigned short test_vec_permx_us(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_permx(vusa, vusb, vucc, 3);
}

vector signed int test_vec_permx_si(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_permx(vsia, vsib, vucc, 4);
}

vector unsigned int test_vec_permx_ui(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_permx(vuia, vuib, vucc, 5);
}

vector signed long long test_vec_permx_sll(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_permx(vslla, vsllb, vucc, 6);
}

vector unsigned long long test_vec_permx_ull(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_permx(vulla, vullb, vucc, 7);
}

vector float test_vec_permx_f(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <4 x float>
  // CHECK-NEXT: ret <4 x float>
  return vec_permx(vfa, vfb, vucc, 0);
}

vector double test_vec_permx_d(void) {
  // CHECK: @llvm.ppc.vsx.xxpermx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-NEXT: bitcast <16 x i8> %{{.*}} to <2 x double>
  // CHECK-NEXT: ret <2 x double>
  return vec_permx(vda, vdb, vucc, 1);
}
