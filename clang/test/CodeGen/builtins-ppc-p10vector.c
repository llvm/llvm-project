// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -target-feature +vsx \
// RUN:   -target-cpu pwr10 -triple powerpc64-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefixes=CHECK-BE,CHECK
// RUN: %clang_cc1 -target-feature +vsx \
// RUN:   -target-cpu pwr10 -triple powerpc64le-unknown-unknown -emit-llvm %s \
// RUN:   -o - | FileCheck %s -check-prefixes=CHECK-LE,CHECK

#include <altivec.h>

vector signed __int128 vi128a;
vector signed char vsca, vscb;
vector unsigned char vuca, vucb, vucc;
vector signed short vssa, vssb;
vector unsigned short vusa, vusb, vusc;
vector signed int vsia, vsib;
vector unsigned int vuia, vuib, vuic;
vector signed long long vslla, vsllb;
vector unsigned long long vulla, vullb, vullc;
vector signed __int128 vsi128a, vsi128b, vsi128c;
vector unsigned __int128 vui128a, vui128b, vui128c;
vector float vfa, vfb;
vector double vda, vdb;
float fa;
double da;
signed int sia;
signed int *iap;
unsigned int uia, uib, *uiap;
signed char *cap;
unsigned char uca;
const unsigned char *ucap;
const signed short *sap;
unsigned short usa;
const unsigned short *usap;
const signed long long *llap;
signed long long llb;
unsigned long long ulla;
const unsigned long long *ullap;

vector signed long long test_vec_mul_sll(void) {
  // CHECK: mul <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_mul(vslla, vsllb);
}

vector unsigned long long test_vec_mul_ull(void) {
  // CHECK: mul <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_mul(vulla, vullb);
}

vector signed int test_vec_div_si(void) {
  // CHECK: sdiv <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_div(vsia, vsib);
}

vector unsigned int test_vec_div_ui(void) {
  // CHECK: udiv <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_div(vuia, vuib);
}

vector signed long long test_vec_div_sll(void) {
  // CHECK: sdiv <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_div(vslla, vsllb);
}

vector unsigned long long test_vec_div_ull(void) {
  // CHECK: udiv <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_div(vulla, vullb);
}

vector unsigned __int128 test_vec_div_u128(void) {
  // CHECK: udiv <1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_div(vui128a, vui128b);
}

vector signed __int128 test_vec_div_s128(void) {
  // CHECK: sdiv <1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_div(vsi128a, vsi128b);
}

vector signed int test_vec_dive_si(void) {
  // CHECK: @llvm.ppc.altivec.vdivesw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_dive(vsia, vsib);
}

vector unsigned int test_vec_dive_ui(void) {
  // CHECK: @llvm.ppc.altivec.vdiveuw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_dive(vuia, vuib);
}

vector signed long long test_vec_dive_sll(void) {
  // CHECK: @llvm.ppc.altivec.vdivesd(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_dive(vslla, vsllb);
}

vector unsigned long long test_vec_dive_ull(void) {
  // CHECK: @llvm.ppc.altivec.vdiveud(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_dive(vulla, vullb);
}

vector unsigned __int128 test_vec_dive_u128(void) {
  // CHECK: @llvm.ppc.altivec.vdiveuq(<1 x i128> %{{.+}}, <1 x i128> %{{.+}})
  // CHECK-NEXT: ret <1 x i128>
  return vec_dive(vui128a, vui128b);
}

vector signed __int128 test_vec_dive_s128(void) {
  // CHECK: @llvm.ppc.altivec.vdivesq(<1 x i128> %{{.+}}, <1 x i128> %{{.+}})
  // CHECK-NEXT: ret <1 x i128>
  return vec_dive(vsi128a, vsi128b);
}

vector signed int test_vec_mulh_si(void) {
  // CHECK: @llvm.ppc.altivec.vmulhsw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_mulh(vsia, vsib);
}

vector unsigned int test_vec_mulh_ui(void) {
  // CHECK: @llvm.ppc.altivec.vmulhuw(<4 x i32> %{{.+}}, <4 x i32> %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_mulh(vuia, vuib);
}

vector signed long long test_vec_mulh_sll(void) {
  // CHECK: @llvm.ppc.altivec.vmulhsd(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_mulh(vslla, vsllb);
}

vector unsigned long long test_vec_mulh_ull(void) {
  // CHECK: @llvm.ppc.altivec.vmulhud(<2 x i64> %{{.+}}, <2 x i64> %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_mulh(vulla, vullb);
}

vector signed int test_vec_mod_si(void) {
  // CHECK: srem <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_mod(vsia, vsib);
}

vector unsigned int test_vec_mod_ui(void) {
  // CHECK: urem <4 x i32>
  // CHECK-NEXT: ret <4 x i32>
  return vec_mod(vuia, vuib);
}

vector signed long long test_vec_mod_sll(void) {
  // CHECK: srem <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_mod(vslla, vsllb);
}

vector unsigned long long test_vec_mod_ull(void) {
  // CHECK: urem <2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_mod(vulla, vullb);
}

vector unsigned char test_xvcvspbf16(vector unsigned char vc) {
  // CHECK-LABEL: @test_xvcvspbf16(
  // CHECK:    [[TMP0:%.*]] = call <16 x i8> @llvm.ppc.vsx.xvcvspbf16(<16 x i8> [[VC:%.*]])
  return __builtin_vsx_xvcvspbf16(vc);
}

vector unsigned char test_xvcvbf16spn(vector unsigned char vc) {
  // CHECK-LABEL: @test_xvcvbf16spn(
  // CHECK:    [[TMP0:%.*]] = call <16 x i8> @llvm.ppc.vsx.xvcvbf16spn(<16 x i8> [[VC:%.*]])
  return __builtin_vsx_xvcvbf16spn(vc);
}

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

vector unsigned char test_vec_stril_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_stril(vuca);
}

vector signed char test_vec_stril_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_stril(vsca);
}

vector unsigned short test_vec_stril_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_stril(vusa);
}

vector signed short test_vec_stril_ss(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_stril(vssa);
}

int test_vec_stril_p_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_stril_p(vuca);
}

int test_vec_stril_p_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_stril_p(vsca);
}

int test_vec_stril_p_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_stril_p(vusa);
}

int test_vec_stril_p_ss(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_stril_p(vssa);
}

vector unsigned char test_vec_stril_p_uc_2(vector unsigned char *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-BE: ret <16 x i8>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-LE: ret <16 x i8>
  for (int i = 0; i < len; i++) {
    if (vec_stril_p(*(ptr + i))) {
      return vec_stril(*(ptr + i));
    }
  }
  return vec_stril(*(ptr));
}

vector signed char test_vec_stril_p_sc_2(vector signed char *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-BE: ret <16 x i8>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-LE: ret <16 x i8>
  for (int i = 0; i < len; i++) {
    if (vec_stril_p(*(ptr + i))) {
      return vec_stril(*(ptr + i));
    }
  }
  return vec_stril(*(ptr));
}

vector unsigned short test_vec_stril_p_us_2(vector unsigned short *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-BE: ret <8 x i16>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-LE: ret <8 x i16>
  for (int i = 0; i < len; i++) {
    if (vec_stril_p(*(ptr + i))) {
      return vec_stril(*(ptr + i));
    }
  }
  return vec_stril(*(ptr));
}

vector signed short test_vec_stril_p_ss_2(vector signed short *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-BE: ret <8 x i16>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-LE: ret <8 x i16>
  for (int i = 0; i < len; i++) {
    if (vec_stril_p(*(ptr + i))) {
      return vec_stril(*(ptr + i));
    }
  }
  return vec_stril(*(ptr));
}

vector unsigned char test_vec_strir_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_strir(vuca);
}

vector signed char test_vec_strir_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_strir(vsca);
}

vector unsigned short test_vec_strir_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_strir(vusa);
}

vector signed short test_vec_strir_ss(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret <8 x i16>
  // CHECK-LE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret <8 x i16>
  return vec_strir(vssa);
}

int test_vec_strir_p_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_strir_p(vuca);
}

int test_vec_strir_p_sc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_strir_p(vsca);
}

int test_vec_strir_p_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_strir_p(vusa);
}

int test_vec_strir_p_ss(void) {
  // CHECK-BE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE-NEXT: ret i32
  // CHECK-LE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE-NEXT: ret i32
  return vec_strir_p(vssa);
}

vector unsigned char test_vec_strir_p_uc_2(vector unsigned char *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-BE: ret <16 x i8>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-LE: ret <16 x i8>
  for (int i = 0; i < len; i++) {
    if (vec_strir_p(*(ptr + i))) {
      return vec_strir(*(ptr + i));
    }
  }
  return vec_strir(*(ptr));
}

vector signed char test_vec_strir_p_sc_2(vector signed char *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstribr.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstribr(<16 x i8> %{{.+}})
  // CHECK-BE: ret <16 x i8>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstribl.p(i32 0, <16 x i8> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstribl(<16 x i8> %{{.+}})
  // CHECK-LE: ret <16 x i8>
  for (int i = 0; i < len; i++) {
    if (vec_strir_p(*(ptr + i))) {
      return vec_strir(*(ptr + i));
    }
  }
  return vec_strir(*(ptr));
}

vector unsigned short test_vec_strir_p_us_2(vector unsigned short *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-BE: ret <8 x i16>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-LE: ret <8 x i16>
  for (int i = 0; i < len; i++) {
    if (vec_strir_p(*(ptr + i))) {
      return vec_strir(*(ptr + i));
    }
  }
  return vec_strir(*(ptr));
}

vector signed short test_vec_strir_p_ss_2(vector signed short *ptr, int len) {
  // CHECK-BE: icmp slt i32
  // CHECK-BE: br i1
  // CHECK-BE: for.body:
  // CHECK-BE: @llvm.ppc.altivec.vstrihr.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-BE: if.then:
  // CHECK-BE: @llvm.ppc.altivec.vstrihr(<8 x i16> %{{.+}})
  // CHECK-BE: ret <8 x i16>
  // CHECK-LE: icmp slt i32
  // CHECK-LE: br i1
  // CHECK-LE: for.body:
  // CHECK-LE: @llvm.ppc.altivec.vstrihl.p(i32 0, <8 x i16> %{{.+}})
  // CHECK-LE: if.then:
  // CHECK-LE: @llvm.ppc.altivec.vstrihl(<8 x i16> %{{.+}})
  // CHECK-LE: ret <8 x i16>
  for (int i = 0; i < len; i++) {
    if (vec_strir_p(*(ptr + i))) {
      return vec_strir(*(ptr + i));
    }
  }
  return vec_strir(*(ptr));
}

unsigned int test_vec_extractm_uc(void) {
  // CHECK: @llvm.ppc.altivec.vextractbm(<16 x i8> %{{.+}})
  // CHECK-NEXT: ret i32
  return vec_extractm(vuca);
}

unsigned int test_vec_extractm_us(void) {
  // CHECK: @llvm.ppc.altivec.vextracthm(<8 x i16> %{{.+}})
  // CHECK-NEXT: ret i32
  return vec_extractm(vusa);
}

unsigned int test_vec_extractm_ui(void) {
  // CHECK: @llvm.ppc.altivec.vextractwm(<4 x i32> %{{.+}})
  // CHECK-NEXT: ret i32
  return vec_extractm(vuia);
}

unsigned int test_vec_extractm_ull(void) {
  // CHECK: @llvm.ppc.altivec.vextractdm(<2 x i64> %{{.+}})
  // CHECK-NEXT: ret i32
  return vec_extractm(vulla);
}

unsigned int test_vec_extractm_u128(void) {
  // CHECK: @llvm.ppc.altivec.vextractqm(<1 x i128> %{{.+}})
  // CHECK-NEXT: ret i32
  return vec_extractm(vui128a);
}

vector unsigned long long test_vcfuged(void) {
  // CHECK: @llvm.ppc.altivec.vcfuged(<2 x i64>
  // CHECK-NEXT: ret <2 x i64>
  return vec_cfuge(vulla, vullb);
}

vector unsigned char test_vec_expandm_uc(void) {
  // CHECK: @llvm.ppc.altivec.vexpandbm(<16 x i8> %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_expandm(vuca);
}

vector unsigned short test_vec_expandm_us(void) {
  // CHECK: @llvm.ppc.altivec.vexpandhm(<8 x i16> %{{.+}})
  // CHECK-NEXT: ret <8 x i16>
  return vec_expandm(vusa);
}

vector unsigned int test_vec_expandm_ui(void) {
  // CHECK: @llvm.ppc.altivec.vexpandwm(<4 x i32> %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_expandm(vuia);
}

vector unsigned long long test_vec_expandm_ull(void) {
  // CHECK: @llvm.ppc.altivec.vexpanddm(<2 x i64> %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_expandm(vulla);
}

vector unsigned __int128 test_vec_expandm_u128(void) {
  // CHECK: @llvm.ppc.altivec.vexpandqm(<1 x i128> %{{.+}})
  // CHECK-NEXT: ret <1 x i128>
  return vec_expandm(vui128a);
}

unsigned long long test_vec_cntm_uc(void) {
  // CHECK: @llvm.ppc.altivec.vcntmbb(<16 x i8> %{{.+}}, i32
  // CHECK-NEXT: ret i64
  return vec_cntm(vuca, 1);
}

unsigned long long test_vec_cntm_us(void) {
  // CHECK: @llvm.ppc.altivec.vcntmbh(<8 x i16> %{{.+}}, i32
  // CHECK-NEXT: ret i64
  return vec_cntm(vusa, 0);
}

unsigned long long test_vec_cntm_ui(void) {
  // CHECK: @llvm.ppc.altivec.vcntmbw(<4 x i32> %{{.+}}, i32
  // CHECK-NEXT: ret i64
  return vec_cntm(vuia, 1);
}

unsigned long long test_vec_cntm_ull(void) {
  // CHECK: @llvm.ppc.altivec.vcntmbd(<2 x i64> %{{.+}}, i32
  // CHECK-NEXT: ret i64
  return vec_cntm(vulla, 0);
}

vector unsigned char test_vec_genbm(void) {
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(ulla);
}

vector unsigned char test_vec_genbm_imm(void) {
  // CHECK: store i64 1
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(1);
}

vector unsigned char test_vec_genbm_imm2(void) {
  // CHECK: store i64 255
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(255);
}

vector unsigned char test_vec_genbm_imm3(void) {
  // CHECK: store i64 65535
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(65535);
}

vector unsigned char test_vec_genbm_imm4(void) {
  // CHECK: store i64 65536
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(65536);
}

vector unsigned char test_vec_genbm_imm5(void) {
  // CHECK: store i64 65546
  // CHECK: @llvm.ppc.altivec.mtvsrbm(i64 %{{.+}})
  // CHECK-NEXT: ret <16 x i8>
  return vec_genbm(65546);
}

vector unsigned short test_vec_genhm(void) {
  // CHECK: @llvm.ppc.altivec.mtvsrhm(i64 %{{.+}})
  // CHECK-NEXT: ret <8 x i16>
  return vec_genhm(ulla);
}

vector unsigned int test_vec_genwm(void) {
  // CHECK: @llvm.ppc.altivec.mtvsrwm(i64 %{{.+}})
  // CHECK-NEXT: ret <4 x i32>
  return vec_genwm(ulla);
}

vector unsigned long long test_vec_gendm(void) {
  // CHECK: @llvm.ppc.altivec.mtvsrdm(i64 %{{.+}})
  // CHECK-NEXT: ret <2 x i64>
  return vec_gendm(ulla);
}

vector unsigned __int128 test_vec_genqm(void) {
  // CHECK: @llvm.ppc.altivec.mtvsrqm(i64 %{{.+}})
  // CHECK-NEXT: ret <1 x i128>
  return vec_genqm(ulla);
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

vector signed int test_vec_replace_elt_si(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 0
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 12
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_replace_elt(vsia, sia, 0);
}

vector unsigned int test_vec_replace_elt_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 4
  // CHECK-BE-NEXT: ret <4 x i32>
  // CHECK-LE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 8
  // CHECK-LE-NEXT: ret <4 x i32>
  return vec_replace_elt(vuia, uia, 1);
}

vector float test_vec_replace_elt_f(void) {
  // CHECK-BE: bitcast float %{{.+}} to i32
  // CHECK-BE-NEXT: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 8
  // CHECK-BE-NEXT: bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK-BE-NEXT: ret <4 x float>
  // CHECK-LE: bitcast float %{{.+}} to i32
  // CHECK-LE-NEXT: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 4
  // CHECK-LE-NEXT: bitcast <4 x i32> %{{.*}} to <4 x float>
  // CHECK-LE-NEXT: ret <4 x float>
  return vec_replace_elt(vfa, fa, 2);
}

vector signed long long test_vec_replace_elt_sll(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 0
  // CHECK-BE-NEXT: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 8
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_replace_elt(vslla, llb, 0);
}

vector unsigned long long test_vec_replace_elt_ull(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 0
  // CHECK-BE-NEXT: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 8
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_replace_elt(vulla, ulla, 0);
}

vector double test_vec_replace_elt_d(void) {
  // CHECK-BE: bitcast double %{{.+}} to i64
  // CHECK-BE-NEXT: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 8
  // CHECK-BE-NEXT: bitcast <2 x i64> %{{.*}} to <2 x double>
  // CHECK-BE-NEXT: ret <2 x double>
  // CHECK-LE: bitcast double %{{.+}} to i64
  // CHECK-LE-NEXT: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 0
  // CHECK-LE-NEXT: bitcast <2 x i64> %{{.*}} to <2 x double>
  // CHECK-LE-NEXT: ret <2 x double>
  return vec_replace_elt(vda, da, 1);
}

vector unsigned char test_vec_replace_unaligned_si(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 6
  // CHECK-BE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 6
  // CHECK-LE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vsia, sia, 6);
}

vector unsigned char test_vec_replace_unaligned_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 8
  // CHECK-BE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 4
  // CHECK-LE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vuia, uia, 8);
}

vector unsigned char test_vec_replace_unaligned_f(void) {
  // CHECK-BE: bitcast float %{{.+}} to i32
  // CHECK-BE-NEXT: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 12
  // CHECK-BE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: bitcast float %{{.+}} to i32
  // CHECK-LE-NEXT: @llvm.ppc.altivec.vinsw(<4 x i32> %{{.+}}, i32 %{{.+}}, i32 0
  // CHECK-LE-NEXT: bitcast <4 x i32> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vfa, fa, 12);
}

vector unsigned char test_vec_replace_unaligned_sll(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 6
  // CHECK-BE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 2
  // CHECK-LE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vslla, llb, 6);
}

vector unsigned char test_vec_replace_unaligned_ull(void) {
  // CHECK-BE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 7
  // CHECK-BE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 1
  // CHECK-LE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vulla, ulla, 7);
}

vector unsigned char test_vec_replace_unaligned_d(void) {
  // CHECK-BE: bitcast double %{{.+}} to i64
  // CHECK-BE-NEXT: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 8
  // CHECK-BE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-BE-NEXT: ret <16 x i8>
  // CHECK-LE: bitcast double %{{.+}} to i64
  // CHECK-LE-NEXT: @llvm.ppc.altivec.vinsd(<2 x i64> %{{.+}}, i64 %{{.+}}, i32 0
  // CHECK-LE-NEXT: bitcast <2 x i64> %{{.*}} to <16 x i8>
  // CHECK-LE-NEXT: ret <16 x i8>
  return vec_replace_unaligned(vda, da, 8);
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

vector unsigned long long test_vec_extractl_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextdubvlx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextdubvrx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extractl(vuca, vucb, uia);
}

vector unsigned long long test_vec_extractl_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextduhvlx(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextduhvrx(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extractl(vusa, vusb, uia);
}

vector unsigned long long test_vec_extractl_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextduwvlx(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextduwvrx(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extractl(vuia, vuib, uia);
}

vector unsigned long long test_vec_extractl_ul(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextddvlx(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextddvrx(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extractl(vulla, vullb, uia);
}

vector unsigned long long test_vec_extracth_uc(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextdubvrx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextdubvlx(<16 x i8> %{{.+}}, <16 x i8> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extracth(vuca, vucb, uia);
}

vector unsigned long long test_vec_extracth_us(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextduhvrx(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextduhvlx(<8 x i16> %{{.+}}, <8 x i16> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extracth(vusa, vusb, uia);
}

vector unsigned long long test_vec_extracth_ui(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextduwvrx(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextduwvlx(<4 x i32> %{{.+}}, <4 x i32> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extracth(vuia, vuib, uia);
}

vector unsigned long long test_vec_extracth_ul(void) {
  // CHECK-BE: @llvm.ppc.altivec.vextddvrx(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32
  // CHECK-BE: [[T1:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T2:%.+]] = bitcast <2 x i64> %{{.*}} to <4 x i32>
  // CHECK-BE: [[T3:%.+]] = call <4 x i32> @llvm.ppc.altivec.vperm(<4 x i32> [[T1]], <4 x i32> [[T2]], <16 x i8> {{.+}})
  // CHECK-BE: [[T4:%.+]] = bitcast <4 x i32> [[T3]] to <2 x i64>
  // CHECK-BE: ret <2 x i64>
  // CHECK-LE: @llvm.ppc.altivec.vextddvlx(<2 x i64> %{{.+}}, <2 x i64> %{{.+}}, i32
  // CHECK-LE-NEXT: ret <2 x i64>
  return vec_extracth(vulla, vullb, uia);
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
  // CHECK-BE-NEXT: [[T2:%.+]] = insertelement <2 x double> poison, double [[T1:%.+]], i32 0
  // CHECK-BE-NEXT: [[T3:%.+]] = shufflevector <2 x double> [[T2:%.+]], <2 x double> poison, <2 x i32> zeroinitialize
  // CHECK-BE-NEXT: ret <2 x double> [[T3:%.+]]
  // CHECK-LE: [[T1:%.+]] = fpext float %{{.+}} to double
  // CHECK-LE-NEXT: [[T2:%.+]] = insertelement <2 x double> poison, double [[T1:%.+]], i32 0
  // CHECK-LE-NEXT: [[T3:%.+]] = shufflevector <2 x double> [[T2:%.+]], <2 x double> poison, <2 x i32> zeroinitialize
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

vector unsigned __int128 test_vec_slq_unsigned (void) {
  // CHECK-LABEL: test_vec_slq_unsigned
  // CHECK: shl <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128> %{{.+}}
  return vec_sl(vui128a, vui128b);
}

vector signed __int128 test_vec_slq_signed (void) {
  // CHECK-LABEL: test_vec_slq_signed
  // CHECK: shl <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128>
  return vec_sl(vi128a, vui128a);
}

vector unsigned __int128 test_vec_srq_unsigned (void) {
  // CHECK-LABEL: test_vec_srq_unsigned
  // CHECK: lshr <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128>
  return vec_sr(vui128a, vui128b);
}

vector signed __int128 test_vec_srq_signed (void) {
  // CHECK-LABEL: test_vec_srq_signed
  // CHECK: lshr <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128>
  return vec_sr(vi128a, vui128a);
}

vector unsigned __int128 test_vec_sraq_unsigned (void) {
  // CHECK-LABEL: test_vec_sraq_unsigned
  // CHECK: ashr <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128>
  return vec_sra(vui128a, vui128b);
}

vector signed __int128 test_vec_sraq_signed (void) {
  // CHECK-LABEL: test_vec_sraq_signed
  // CHECK: ashr <1 x i128> %{{.+}}, %{{.+}}
  // CHECK: ret <1 x i128>
  return vec_sra(vi128a, vui128a);
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

vector unsigned __int128 test_vec_mule_u128(void) {
  // CHECK-BE: @llvm.ppc.altivec.vmuleud(<2 x i64>
  // CHECK-BE-NEXT: ret <1 x i128>
  // CHECK-LE: @llvm.ppc.altivec.vmuloud(<2 x i64>
  // CHECK-LE-NEXT: ret <1 x i128>
  return vec_mule(vulla, vullb);
}

vector signed __int128 test_vec_mule_s128(void) {
  // CHECK-BE: @llvm.ppc.altivec.vmulesd(<2 x i64>
  // CHECK-BE-NEXT: ret <1 x i128>
  // CHECK-LE: @llvm.ppc.altivec.vmulosd(<2 x i64>
  // CHECK-LE-NEXT: ret <1 x i128>
  return vec_mule(vslla, vsllb);
}

vector unsigned __int128 test_vec_mulo_u128(void) {
  // CHECK-BE: @llvm.ppc.altivec.vmuloud(<2 x i64>
  // CHECK-BE-NEXT: ret <1 x i128>
  // CHECK-LE: @llvm.ppc.altivec.vmuleud(<2 x i64>
  // CHECK-LE-NEXT: ret <1 x i128>
  return vec_mulo(vulla, vullb);
}

vector signed __int128 test_vec_mulo_s128(void) {
  // CHECK-BE: @llvm.ppc.altivec.vmulosd(<2 x i64>
  // CHECK-BE-NEXT: ret <1 x i128>
  // CHECK-LE: @llvm.ppc.altivec.vmulesd(<2 x i64>
  // CHECK-LE-NEXT: ret <1 x i128>
  return vec_mulo(vslla, vsllb);
}

vector unsigned __int128 test_vec_msumc_u128(void) {
  // CHECK: @llvm.ppc.altivec.vmsumcud(<2 x i64>
  // CHECK-NEXT: ret <1 x i128>
  return vec_msumc(vulla, vullb, vui128a);
}

vector signed __int128 test_vec_xl_sext_i8(void) {
  // CHECK: load i8
  // CHECK: sext i8
  // CHECK: ret <1 x i128>
  return vec_xl_sext(llb, cap);
}

vector signed __int128 test_vec_xl_sext_i16(void) {
  // CHECK: load i16
  // CHECK: sext i16
  // CHECK: ret <1 x i128>
  return vec_xl_sext(llb, sap);
}

vector signed __int128 test_vec_xl_sext_i32(void) {
  // CHECK: load i32
  // CHECK: sext i32
  // CHECK: ret <1 x i128>
  return vec_xl_sext(llb, iap);
}

vector signed __int128 test_vec_xl_sext_i64(void) {
  // CHECK: load i64
  // CHECK: sext i64
  // CHECK: ret <1 x i128>
  return vec_xl_sext(llb, llap);
}

vector unsigned __int128 test_vec_xl_zext_i8(void) {
  // CHECK: load i8
  // CHECK: zext i8
  // CHECK: ret <1 x i128>
  return vec_xl_zext(llb, ucap);
}

vector unsigned __int128 test_vec_xl_zext_i16(void) {
  // CHECK: load i16
  // CHECK: zext i16
  // CHECK: ret <1 x i128>
  return vec_xl_zext(llb, usap);
}

vector unsigned __int128 test_vec_xl_zext_i32(void) {
  // CHECK: load i32
  // CHECK: zext i32
  // CHECK: ret <1 x i128>
  return vec_xl_zext(llb, uiap);
}

vector unsigned __int128 test_vec_xl_zext_i64(void) {
  // CHECK: load i64
  // CHECK: zext i64
  // CHECK: ret <1 x i128>
  return vec_xl_zext(llb, ullap);
}

vector signed __int128 test_vec_signextq_s128(void) {
    // CHECK: @llvm.ppc.altivec.vextsd2q(<2 x i64>
    // CHECK-NEXT: ret <1 x i128>
    return vec_signextq(vslla);
}

vector unsigned __int128 test_vec_mod_u128(void) {
    // CHECK: urem <1 x i128>
    // CHECK-NEXT: ret <1 x i128>
    return vec_mod(vui128a, vui128b);
}

vector signed __int128 test_vec_mod_s128(void) {
    // CHECK: srem <1 x i128>
    // CHECK-NEXT: ret <1 x i128>
    return vec_mod(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmpeq_s128(void) {
  // CHECK-LABEL: @test_vec_cmpeq_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpeq(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmpeq_u128(void) {
  // CHECK-LABEL: @test_vec_cmpeq_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpeq(vui128a, vui128b);
}

vector bool __int128 test_vec_cmpne_s128(void) {
  // CHECK-LABEL: @test_vec_cmpne_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128>
  // CHECK-NEXT: %neg.i = xor <1 x i128> %4, <i128 -1>
  // CHECK-NEXT: ret <1 x i128> %neg.i
  return vec_cmpne(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmpne_u128(void) {
  // CHECK-LABEL: @test_vec_cmpne_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpequq(<1 x i128>
  // CHECK-NEXT: %neg.i = xor <1 x i128> %4, <i128 -1>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpne(vui128a, vui128b);
}

vector bool __int128 test_vec_cmpgt_s128(void) {
  // CHECK-LABEL: @test_vec_cmpgt_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpgt(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmpgt_u128(void) {
  // CHECK-LABEL: @test_vec_cmpgt_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpgt(vui128a, vui128b);
}

vector bool __int128 test_vec_cmplt_s128(void) {
  // CHECK-LABEL: @test_vec_cmplt_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmplt(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmplt_u128(void) {
  // CHECK-LABEL: @test_vec_cmplt_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmplt(vui128a, vui128b);
}

vector bool __int128 test_vec_cmpge_s128(void) {
  // CHECK-LABEL: @test_vec_cmpge_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128>
  // CHECK-NEXT: %neg.i = xor <1 x i128> %6, <i128 -1>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpge(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmpge_u128(void) {
  // CHECK-LABEL: @test_vec_cmpge_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128>
  // CHECK-NEXT: %neg.i = xor <1 x i128> %6, <i128 -1>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmpge(vui128a, vui128b);
}

vector bool __int128 test_vec_cmple_s128(void) {
  // CHECK-LABEL: @test_vec_cmple_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtsq(<1 x i128>
  // CHECK-NEXT: %neg.i.i = xor <1 x i128> %8, <i128 -1>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmple(vsi128a, vsi128b);
}

vector bool __int128 test_vec_cmple_u128(void) {
  // CHECK-LABEL: @test_vec_cmple_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vcmpgtuq(<1 x i128>
  // CHECK-NEXT: %neg.i.i = xor <1 x i128> %8, <i128 -1>
  // CHECK-NEXT: ret <1 x i128>
  return vec_cmple(vui128a, vui128b);
}

int test_vec_any_eq_u128(void) {
  // CHECK-LABEL: @test_vec_any_eq_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_eq(vui128a, vui128b);
}

int test_vec_any_eq_s128(void) {
  // CHECK-LABEL: @test_vec_any_eq_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_eq(vsi128a, vsi128b);
}

int test_vec_any_ne_s128(void) {
  // CHECK-LABEL: @test_vec_any_ne_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_ne(vsi128a, vsi128b);
}

int test_vec_any_ne_u128(void) {
  // CHECK-LABEL: @test_vec_any_ne_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_ne(vui128a, vui128b);
}

int test_vec_any_lt_s128(void) {
  // CHECK-LABEL: @test_vec_any_lt_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_lt(vsi128a, vsi128b);
}

int test_vec_any_lt_u128(void) {
  // CHECK-LABEL: @test_vec_any_lt_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_lt(vui128a, vui128b);
}

int test_vec_any_gt_s128(void) {
  // CHECK-LABEL: @test_vec_any_gt_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_gt(vsi128a, vsi128b);
}

int test_vec_any_gt_u128(void) {
  // CHECK-LABEL: @test_vec_any_gt_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 1, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_gt(vui128a, vui128b);
}

int test_vec_any_le_s128(void) {
  // CHECK-LABEL: @test_vec_any_le_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_le(vsi128a, vsi128b);
}

int test_vec_any_le_u128(void) {
  // CHECK-LABEL: @test_vec_any_le_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_le(vui128a, vui128b);
}

int test_vec_any_ge_s128(void) {
  // CHECK-LABEL: @test_vec_any_ge_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_ge(vsi128a, vsi128b);
}

int test_vec_any_ge_u128(void) {
  // CHECK-LABEL: @test_vec_any_ge_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 3, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_any_ge(vui128a, vui128b);
}

int test_vec_all_eq_s128(void) {
  // CHECK-LABEL: @test_vec_all_eq_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_eq(vsi128a, vsi128b);
}

int test_vec_all_eq_u128(void) {
  // CHECK-LABEL: @test_vec_all_eq_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_eq(vui128a, vui128b);
}

int test_vec_all_ne_s128(void) {
  // CHECK-LABEL: @test_vec_all_ne_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_ne(vsi128a, vsi128b);
}

int test_vec_all_ne_u128(void) {
  // CHECK-LABEL: @test_vec_all_ne_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpequq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_ne(vui128a, vui128b);
}

int test_vec_all_lt_s128(void) {
  // CHECK-LABEL: @test_vec_all_lt_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_lt(vsi128a, vsi128b);
}

int test_vec_all_lt_u128(void) {
  // CHECK-LABEL: @test_vec_all_lt_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK: ret i32
  return vec_all_lt(vui128a, vui128b);
}

int test_vec_all_gt_s128(void) {
  // CHECK-LABEL: @test_vec_all_gt_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_gt(vsi128a, vsi128b);
}

int test_vec_all_gt_u128(void) {
  // CHECK-LABEL: @test_vec_all_gt_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 2, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_gt(vui128a, vui128b);
}

int test_vec_all_le_s128(void) {
  // CHECK-LABEL: @test_vec_all_le_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_le(vsi128a, vsi128b);
}

int test_vec_all_le_u128(void) {
  // CHECK-LABEL: @test_vec_all_le_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_le(vui128a, vui128b);
}

int test_vec_all_ge_s128(void) {
  // CHECK-LABEL: @test_vec_all_ge_s128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtsq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_ge(vsi128a, vsi128b);
}

int test_vec_all_ge_u128(void) {
  // CHECK-LABEL: @test_vec_all_ge_u128(
  // CHECK: call i32 @llvm.ppc.altivec.vcmpgtuq.p(i32 0, <1 x i128> %2, <1 x i128> %3)
  // CHECK-NEXT: ret i32
  return vec_all_ge(vui128a, vui128b);
}

vector signed __int128 test_vec_rl_s128(void) {
  // CHECK-LABEL: @test_vec_rl_s128(
  // CHECK: sub <1 x i128>
  // CHECK-NEXT: lshr <1 x i128>
  // CHECK-NEXT: or <1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rl(vsi128a, vsi128b);
}

vector unsigned __int128 test_vec_rl_u128(void) {
  // CHECK-LABEL: @test_vec_rl_u128(
  // CHECK: sub <1 x i128>
  // CHECK: lshr <1 x i128>
  // CHECK: or <1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rl(vui128a, vui128b);
}

vector signed __int128 test_vec_rlnm_s128(void) {
  // CHECK-LABEL: @test_vec_rlnm_s128(
  // CHECK-LE: %shuffle.i = shufflevector <16 x i8> %7, <16 x i8> %8, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK-BE: %shuffle.i = shufflevector <16 x i8> %7, <16 x i8> %8, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 31, i32 30, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vrlqnm(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rlnm(vsi128a, vsi128b, vsi128c);
}

vector unsigned __int128 test_vec_rlnm_u128(void) {
  // CHECK-LABEL: @test_vec_rlnm_u128(
  // CHECK-LE:  %shuffle.i = shufflevector <16 x i8> %7, <16 x i8> %8, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 16, i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK-BE: %shuffle.i = shufflevector <16 x i8> %7, <16 x i8> %8, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 31, i32 30, i32 15, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vrlqnm(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rlnm(vui128a, vui128b, vui128c);
}

vector signed __int128 test_vec_rlmi_s128(void) {
  // CHECK-LABEL: @test_vec_rlmi_s128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vrlqmi(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rlmi(vsi128a, vsi128b, vsi128c);
}

vector unsigned __int128 test_vec_rlmi_u128(void) {
  // CHECK-LABEL: @test_vec_rlmi_u128(
  // CHECK: call <1 x i128> @llvm.ppc.altivec.vrlqmi(<1 x i128>
  // CHECK-NEXT: ret <1 x i128>
  return vec_rlmi(vui128a, vui128b, vui128c);
}
