// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +vsx \
// RUN:   -target-cpu pwr10 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefixes=CHECK-BE,CHECK

// RUN: %clang_cc1 -target-feature +vsx \
// RUN:   -target-cpu pwr10 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefixes=CHECK-LE,CHECK

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
unsigned int uia, uib;
unsigned char uca;
unsigned short usa;
unsigned long long ulla;

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

vector signed char test_vec_blend_sc(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvb(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8>
  // CHECK-NEXT: ret <16 x i8>
  return vec_blendv(vsca, vscb, vucc);
}

vector unsigned char test_vec_blend_uc(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvb(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, <16 x i8>
  // CHECK-NEXT: ret <16 x i8>
  return vec_blendv(vuca, vucb, vucc);
}

vector signed short test_vec_blend_ss(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvh(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_blendv(vssa, vssb, vusc);
}

vector unsigned short test_vec_blend_us(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvh(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, <8 x i16>
  // CHECK-NEXT: ret <8 x i16>
  return vec_blendv(vusa, vusb, vusc);
}

vector signed int test_vec_blend_si(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_blendv(vsia, vsib, vuic);
}

vector unsigned int test_vec_blend_ui(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_blendv(vuia, vuib, vuic);
}

vector signed long long test_vec_blend_sll(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvd(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_blendv(vslla, vsllb, vullc);
}

vector unsigned long long test_vec_blend_ull(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvd(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_blendv(vulla, vullb, vullc);
}

vector float test_vec_blend_f(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, <4 x i32>
  // CHECK-NEXT: bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK-NEXT: ret <4 x float>
  return vec_blendv(vfa, vfb, vuic);
}

vector double test_vec_blend_d(void) {
  // CHECK: @llvm.ppc.vsx.xxblendvd(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, <2 x i64>
  // CHECK-NEXT: bitcast <2 x i64> %{{.*}} to <2 x double>
  // CHECK-NEXT: ret <2 x double>
  return vec_blendv(vda, vdb, vullc);
}

vector unsigned char test_vec_insertl_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsblx(<16 x i8> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsbrx(<16 x i8> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_insertl(uca, vuca, uia);
}

vector unsigned short test_vec_insertl_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinshlx(<8 x i16> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vinshrx(<8 x i16> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_insertl(usa, vusa, uia);
}

vector unsigned int test_vec_insertl_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinswlx(<4 x i32> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinswrx(<4 x i32> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_insertl(uib, vuia, uia);
}

vector unsigned long long test_vec_insertl_ul(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsdlx(<2 x i64> %{{.+}}, i64 %{{.+}}, i64
  // CHECK-BE-NEXT: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vinsdrx(<2 x i64> %{{.+}}, i64 %{{.+}}, i64
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_insertl(ulla, vulla, uia);
}

vector unsigned char test_vec_insertl_ucv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsbvlx(<16 x i8> %{{.+}}, i32 %{{.+}}, <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsbvrx(<16 x i8> %{{.+}}, i32 %{{.+}}, <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_insertl(vuca, vucb, uia);
}

vector unsigned short test_vec_insertl_usv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinshvlx(<8 x i16> %{{.+}}, i32 %{{.+}}, <8 x i16>
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vinshvrx(<8 x i16> %{{.+}}, i32 %{{.+}}, <8 x i16>
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_insertl(vusa, vusb, uia);
}

vector unsigned int test_vec_insertl_uiv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinswvlx(<4 x i32> %{{.+}}, i32 %{{.+}}, <4 x i32>
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinswvrx(<4 x i32> %{{.+}}, i32 %{{.+}}, <4 x i32>
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_insertl(vuia, vuib, uia);
}

vector unsigned char test_vec_inserth_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsbrx(<16 x i8> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsblx(<16 x i8> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_inserth(uca, vuca, uia);
}

vector unsigned short test_vec_inserth_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinshrx(<8 x i16> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vinshlx(<8 x i16> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_inserth(usa, vusa, uia);
}

vector unsigned int test_vec_inserth_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinswrx(<4 x i32> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinswlx(<4 x i32> %{{.+}}, i32 %{{.+}}, i32
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_inserth(uib, vuia, uia);
}

vector unsigned long long test_vec_inserth_ul(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsdrx(<2 x i64> %{{.+}}, i64 %{{.+}}, i64
  // CHECK-BE-NEXT: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vinsdlx(<2 x i64> %{{.+}}, i64 %{{.+}}, i64
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_inserth(ulla, vulla, uia);
}

vector unsigned char test_vec_inserth_ucv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsbvrx(<16 x i8> %{{.+}}, i32 %{{.+}}, <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsbvlx(<16 x i8> %{{.+}}, i32 %{{.+}}, <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_inserth(vuca, vucb, uia);
}

vector unsigned short test_vec_inserth_usv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinshvrx(<8 x i16> %{{.+}}, i32 %{{.+}}, <8 x i16>
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vinshvlx(<8 x i16> %{{.+}}, i32 %{{.+}}, <8 x i16>
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_inserth(vusa, vusb, uia);
}

vector unsigned int test_vec_inserth_uiv(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinswvrx(<4 x i32> %{{.+}}, i32 %{{.+}}, <4 x i32>
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinswvlx(<4 x i32> %{{.+}}, i32 %{{.+}}, <4 x i32>
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_inserth(vuia, vuib, uia);
}

vector signed int test_vec_vec_splati_si(void) {
  // CHECK: ret <4 x i32> <i32 -17, i32 -17, i32 -17, i32 -17>
  return vec_splati(-17);
}

vector unsigned int test_vec_vec_splati_ui(void) {
  // CHECK: ret <4 x i32> <i32 16, i32 16, i32 16, i32 16>
  return vec_splati(16U);
}

vector float test_vec_vec_splati_f(void) {
  // CHECK: ret <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
  return vec_splati(1.0f);
}

vector double test_vec_vec_splatid(void) {
  // CHECK-BE: [[T1:%.+]] = fpext float %{{.+}} to double
  // CHECK-BE-NEXT: [[T2:%.+]] = insertelement <2 x double> undef, double [[T1:%.+]], i32 0
  // CHECK-BE-NEXT: [[T3:%.+]] = shufflevector <2 x double> [[T2:%.+]], <2 x double> undef, <2 x i32> zeroinitialize
  // CHECK-BE-NEXT: ret <2 x double> [[T3:%.+]]
  // CHECK-LE: [[T1:%.+]] = fpext float %{{.+}} to double
  // CHECK-LE-NEXT: [[T2:%.+]] = insertelement <2 x double> undef, double [[T1:%.+]], i32 0
  // CHECK-LE-NEXT: [[T3:%.+]] = shufflevector <2 x double> [[T2:%.+]], <2 x double> undef, <2 x i32> zeroinitialize
  // CHECK-LE-NEXT: ret <2 x double> [[T3:%.+]]
  return vec_splatid(1.0);
}

vector signed int test_vec_vec_splati_ins_si(void) {
  // CHECK-BE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 %{{.+}}
  // CHECK-BE:  [[T1:%.+]] = add i32 2, %{{.+}}
  // CHECK-BE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T1]]
  // CHECK-BE: ret <4 x i32>
  // CHECK-LE:  [[T1:%.+]] = sub i32 1, %{{.+}}
  // CHECK-LE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T1]]
  // CHECK-LE:  [[T2:%.+]] = sub i32 3, %{{.+}}
  // CHECK-LE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T2]]
  // CHECK-LE: ret <4 x i32>
  return vec_splati_ins(vsia, 0, -17);
}

vector unsigned int test_vec_vec_splati_ins_ui(void) {
  // CHECK-BE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 %{{.+}}
  // CHECK-BE:  [[T1:%.+]] = add i32 2, %{{.+}}
  // CHECK-BE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T1]]
  // CHECK-BE: ret <4 x i32>
  // CHECK-LE:  [[T1:%.+]] = sub i32 1, %{{.+}}
  // CHECK-LE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T1]]
  // CHECK-LE:  [[T2:%.+]] = sub i32 3, %{{.+}}
  // CHECK-LE: insertelement <4 x i32> %{{.+}}, i32 %{{.+}}, i32 [[T2]]
  // CHECK-LE: ret <4 x i32>
  return vec_splati_ins(vuia, 1, 16U);
}

vector float test_vec_vec_splati_ins_f(void) {
  // CHECK-BE: insertelement <4 x float> %{{.+}}, float %{{.+}}, i32 %{{.+}}
  // CHECK-BE:  [[T1:%.+]] = add i32 2, %{{.+}}
  // CHECK-BE: insertelement <4 x float> %{{.+}}, float %{{.+}}, i32 [[T1]]
  // CHECK-BE: ret <4 x float>
  // CHECK-LE:  [[T1:%.+]] = sub i32 1, %{{.+}}
  // CHECK-LE: insertelement <4 x float> %{{.+}}, float %{{.+}}, i32 [[T1]]
  // CHECK-LE:  [[T2:%.+]] = sub i32 3, %{{.+}}
  // CHECK-LE: insertelement <4 x float> %{{.+}}, float %{{.+}}, i32 [[T2]]
  // CHECK-LE: ret <4 x float>
  return vec_splati_ins(vfa, 0, 1.0f);
}

void test_vec_xst_trunc_sc(vector signed __int128 __a, signed long long __b,
                           signed char *__c) {
  // CHECK: store i8 %{{.+}}, i8* %{{.+}}, align 1
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_uc(vector unsigned __int128 __a, signed long long __b,
                           unsigned char *__c) {
  // CHECK: store i8 %{{.+}}, i8* %{{.+}}, align 1
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_ss(vector signed __int128 __a, signed long long __b,
                           signed short *__c) {
  // CHECK: store i16 %{{.+}}, i16* %{{.+}}, align 2
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_us(vector unsigned __int128 __a, signed long long __b,
                           unsigned short *__c) {
  // CHECK: store i16 %{{.+}}, i16* %{{.+}}, align 2
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_si(vector signed __int128 __a, signed long long __b,
                           signed int *__c) {
  // CHECK: store i32 %{{.+}}, i32* %{{.+}}, align 4
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_ui(vector unsigned __int128 __a, signed long long __b,
                           unsigned int *__c) {
  // CHECK: store i32 %{{.+}}, i32* %{{.+}}, align 4
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_sll(vector signed __int128 __a, signed long long __b,
                            signed long long *__c) {
  // CHECK: store i64 %{{.+}}, i64* %{{.+}}, align 8
  vec_xst_trunc(__a, __b, __c);
}

void test_vec_xst_trunc_ull(vector unsigned __int128 __a, signed long long __b,
                            unsigned long long *__c) {
  // CHECK: store i64 %{{.+}}, i64* %{{.+}}, align 8
  vec_xst_trunc(__a, __b, __c);
}

int test_vec_test_lsbb_all_ones(void) {
  // CHECK: @llvm.ppc.vsx.xvtlsbb(<16 x i8> %{{.+}}, i32 1
  // CHECK-NEXT: ret i32
  return vec_test_lsbb_all_ones(vuca);
}

int test_vec_test_lsbb_all_zeros(void) {
  // CHECK: @llvm.ppc.vsx.xvtlsbb(<16 x i8> %{{.+}}, i32 0
  // CHECK-NEXT: ret i32
  return vec_test_lsbb_all_zeros(vuca);
}
