// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z17 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z17 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed char vsc1;
volatile vector signed char vsc2;
volatile vector signed short vss;
volatile vector signed short vss1;
volatile vector signed short vss2;
volatile vector signed int vsi;
volatile vector signed int vsi1;
volatile vector signed int vsi2;
volatile vector signed long long vsl;
volatile vector signed long long vsl1;
volatile vector signed long long vsl2;
volatile vector signed __int128 vslll;
volatile vector signed __int128 vslll1;
volatile vector signed __int128 vslll2;
volatile vector unsigned char vuc;
volatile vector unsigned char vuc1;
volatile vector unsigned char vuc2;
volatile vector unsigned short vus;
volatile vector unsigned short vus1;
volatile vector unsigned short vus2;
volatile vector unsigned int vui;
volatile vector unsigned int vui1;
volatile vector unsigned int vui2;
volatile vector unsigned long long vul;
volatile vector unsigned long long vul1;
volatile vector unsigned long long vul2;
volatile vector unsigned __int128 vulll;
volatile vector unsigned __int128 vulll1;
volatile vector unsigned __int128 vulll2;
volatile vector bool char vbc;
volatile vector bool char vbc1;
volatile vector bool char vbc2;
volatile vector bool short vbs;
volatile vector bool short vbs1;
volatile vector bool short vbs2;
volatile vector bool int vbi;
volatile vector bool int vbi1;
volatile vector bool int vbi2;
volatile vector bool long long vbl;
volatile vector bool long long vbl1;
volatile vector bool long long vbl2;
volatile vector bool __int128 vblll;
volatile vector bool __int128 vblll1;
volatile vector bool __int128 vblll2;
volatile vector float vf;
volatile vector float vf1;
volatile vector double vd;
volatile vector double vd1;

volatile int idx;
int cc;

void test_core(void) {
  // CHECK-ASM-LABEL: test_core

  vuc = vec_gen_element_masks_8(vus);
  // CHECK: call <16 x i8> @llvm.s390.vgemb(<8 x i16> %{{.*}})
  // CHECK-ASM: vgemb
  vus = vec_gen_element_masks_16(vuc);
  // CHECK: call <8 x i16> @llvm.s390.vgemh(<16 x i8> %{{.*}})
  // CHECK-ASM: vgemh
  vui = vec_gen_element_masks_32(vuc);
  // CHECK: call <4 x i32> @llvm.s390.vgemf(<16 x i8> %{{.*}})
  // CHECK-ASM: vgemf
  vul = vec_gen_element_masks_64(vuc);
  // CHECK: call <2 x i64> @llvm.s390.vgemg(<16 x i8> %{{.*}})
  // CHECK-ASM: vgemg
  vulll = vec_gen_element_masks_128(vuc);
  // CHECK: call i128 @llvm.s390.vgemq(<16 x i8> %{{.*}})
  // CHECK-ASM: vgemq

  vsc = vec_blend(vsc, vsc1, vsc2);
  // CHECK-ASM: vblendb
  vbc = vec_blend(vbc, vbc1, vsc);
  // CHECK-ASM: vblendb
  vuc = vec_blend(vuc, vuc1, vsc);
  // CHECK-ASM: vblendb
  vss = vec_blend(vss, vss1, vss2);
  // CHECK-ASM: vblendh
  vbs = vec_blend(vbs, vbs1, vss);
  // CHECK-ASM: vblendh
  vus = vec_blend(vus, vus1, vss);
  // CHECK-ASM: vblendh
  vsi = vec_blend(vsi, vsi1, vsi2);
  // CHECK-ASM: vblendf
  vbi = vec_blend(vbi, vbi1, vsi);
  // CHECK-ASM: vblendf
  vui = vec_blend(vui, vui1, vsi);
  // CHECK-ASM: vblendf
  vsl = vec_blend(vsl, vsl1, vsl2);
  // CHECK-ASM: vblendg
  vul = vec_blend(vul, vul1, vsl);
  // CHECK-ASM: vblendg
  vbl = vec_blend(vbl, vbl1, vsl);
  // CHECK-ASM: vblendg
  vslll = vec_blend(vslll, vslll1, vslll2);
  // CHECK-ASM: vblendq
  vblll = vec_blend(vblll, vblll1, vslll);
  // CHECK-ASM: vblendq
  vulll = vec_blend(vulll, vulll1, vslll);
  // CHECK-ASM: vblendq
  vf = vec_blend(vf, vf1, vsi);
  // CHECK-ASM: vblendf
  vd = vec_blend(vd, vd1, vsl);
  // CHECK-ASM: vblendg

  vslll = vec_unpackh(vsl);
  // CHECK: call i128 @llvm.s390.vuphg(<2 x i64> %{{.*}})
  // CHECK-ASM: vuphg
  vulll = vec_unpackh(vul);
  // CHECK: call i128 @llvm.s390.vuplhg(<2 x i64> %{{.*}})
  // CHECK-ASM: vuplhg
  vslll = vec_unpackl(vsl);
  // CHECK: call i128 @llvm.s390.vuplg(<2 x i64> %{{.*}})
  // CHECK-ASM: vuplg
  vulll = vec_unpackl(vul);
  // CHECK: call i128 @llvm.s390.vupllg(<2 x i64> %{{.*}})
  // CHECK-ASM: vupllg
}

void test_compare(void) {
  // CHECK-ASM-LABEL: test_compare

  vblll = vec_cmpeq(vslll, vslll1);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq
  vblll = vec_cmpeq(vulll, vulll1);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq
  vblll = vec_cmpeq(vblll, vblll1);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq

  vblll = vec_cmpge(vslll, vslll1);
  // CHECK: icmp sge <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmpge(vulll, vulll1);
  // CHECK: icmp uge <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmpgt(vslll, vslll1);
  // CHECK: icmp sgt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmpgt(vulll, vulll1);
  // CHECK: icmp ugt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmple(vslll, vslll1);
  // CHECK: icmp sle <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmple(vulll, vulll1);
  // CHECK: icmp ule <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmplt(vslll, vslll1);
  // CHECK: icmp slt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmplt(vulll, vulll1);
  // CHECK: icmp ult <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  idx = vec_all_eq(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_eq(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_eq(vblll, vblll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_all_ne(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_ne(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_ne(vblll, vblll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_all_ge(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_ge(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_gt(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_gt(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_le(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_le(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_lt(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_lt(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_eq(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_eq(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_eq(vblll, vblll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_any_ne(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_ne(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_ne(vblll, vblll1);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_any_ge(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_ge(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_gt(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_gt(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_le(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_le(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_lt(vslll, vslll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_lt(vulll, vulll1);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs
}

void test_integer(void) {
  // CHECK-ASM-LABEL: test_integer

  vulll = vec_cntlz(vulll);
  // CHECK: call range(i128 0, 129) i128 @llvm.ctlz.i128(i128 %{{.*}}, i1 false)
  // CHECK-ASM: vclzq
  vulll = vec_cnttz(vulll);
  // CHECK: call range(i128 0, 129) i128 @llvm.cttz.i128(i128 %{{.*}}, i1 false)
  // CHECK-ASM: vctzq

  vslll = vec_abs(vslll);
  // CHECK-ASM: vlpq

  vslll = vec_avg(vslll, vslll1);
  // CHECK: call i128 @llvm.s390.vavgq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vavgq
  vulll = vec_avg(vulll, vulll1);
  // CHECK: call i128 @llvm.s390.vavglq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vavglq

  vsc = vec_evaluate(vsc, vsc1, vsc2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsc = vec_evaluate(vsc, vsc1, vsc2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vuc = vec_evaluate(vuc, vuc1, vuc2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vuc = vec_evaluate(vuc, vuc1, vuc2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbc = vec_evaluate(vbc, vbc1, vbc2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbc = vec_evaluate(vbc, vbc1, vbc2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vss = vec_evaluate(vss, vss1, vss2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vss = vec_evaluate(vss, vss1, vss2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vus = vec_evaluate(vus, vus1, vus2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vus = vec_evaluate(vus, vus1, vus2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbs = vec_evaluate(vbs, vbs1, vbs2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbs = vec_evaluate(vbs, vbs1, vbs2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vsi = vec_evaluate(vsi, vsi1, vsi2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsi = vec_evaluate(vsi, vsi1, vsi2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vui = vec_evaluate(vui, vui1, vui2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vui = vec_evaluate(vui, vui1, vui2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbi = vec_evaluate(vbi, vbi1, vbi2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbi = vec_evaluate(vbi, vbi1, vbi2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vsl = vec_evaluate(vsl, vsl1, vsl2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsl = vec_evaluate(vsl, vsl1, vsl2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vul = vec_evaluate(vul, vul1, vul2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vul = vec_evaluate(vul, vul1, vul2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbl = vec_evaluate(vbl, vbl1, vbl2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbl = vec_evaluate(vbl, vbl1, vbl2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vslll = vec_evaluate(vslll, vslll1, vslll2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vslll = vec_evaluate(vslll, vslll1, vslll2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vulll = vec_evaluate(vulll, vulll1, vulll2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vulll = vec_evaluate(vulll, vulll1, vulll2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vblll = vec_evaluate(vblll, vblll1, vblll2, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vblll = vec_evaluate(vblll, vblll1, vblll2, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval

  vslll = vec_max(vslll, vslll1);
  // CHECK-ASM: vmxq
  vulll = vec_max(vulll, vulll1);
  // CHECK-ASM: vmxlq
  vslll = vec_min(vslll, vslll1);
  // CHECK-ASM: vmnq
  vulll = vec_min(vulll, vulll1);
  // CHECK-ASM: vmnlq

  vsl = vec_mladd(vsl, vsl1, vsl2);
  // CHECK-ASM: vmalg
  vsl = vec_mladd(vul, vsl, vsl1);
  // CHECK-ASM: vmalg
  vsl = vec_mladd(vsl, vul, vul1);
  // CHECK-ASM: vmalg
  vul = vec_mladd(vul, vul1, vul2);
  // CHECK-ASM: vmalg
  vslll = vec_mladd(vslll, vslll1, vslll2);
  // CHECK-ASM: vmalq
  vslll = vec_mladd(vulll, vslll, vslll1);
  // CHECK-ASM: vmalq
  vslll = vec_mladd(vslll, vulll, vulll1);
  // CHECK-ASM: vmalq
  vulll = vec_mladd(vulll, vulll1, vulll2);
  // CHECK-ASM: vmalq

  vsl = vec_mhadd(vsl, vsl1, vsl2);
  // CHECK: call <2 x i64> @llvm.s390.vmahg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmahg
  vul = vec_mhadd(vul, vul1, vul2);
  // CHECK: call <2 x i64> @llvm.s390.vmalhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmalhg
  vslll = vec_mhadd(vslll, vslll1, vslll2);
  // CHECK: call i128 @llvm.s390.vmahq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmahq
  vulll = vec_mhadd(vulll, vulll1, vulll2);
  // CHECK: call i128 @llvm.s390.vmalhq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmalhq

  vslll = vec_meadd(vsl, vsl1, vslll);
  // CHECK: call i128 @llvm.s390.vmaeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaeg
  vulll = vec_meadd(vul, vul1, vulll);
  // CHECK: call i128 @llvm.s390.vmaleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaleg

  vslll = vec_moadd(vsl, vsl1, vslll);
  // CHECK: call i128 @llvm.s390.vmaog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaog
  vulll = vec_moadd(vul, vul1, vulll);
  // CHECK: call i128 @llvm.s390.vmalog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmalog

  vsl = vec_mulh(vsl, vsl1);
  // CHECK: call <2 x i64> @llvm.s390.vmhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmhg
  vul = vec_mulh(vul, vul1);
  // CHECK: call <2 x i64> @llvm.s390.vmlhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmlhg
  vslll = vec_mulh(vslll, vslll1);
  // CHECK: call i128 @llvm.s390.vmhq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmhq
  vulll = vec_mulh(vulll, vulll1);
  // CHECK: call i128 @llvm.s390.vmlhq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmlhq

  vslll = vec_mule(vsl, vsl1);
  // CHECK: call i128 @llvm.s390.vmeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmeg
  vulll = vec_mule(vul, vul1);
  // CHECK: call i128 @llvm.s390.vmleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmleg

  vslll = vec_mulo(vsl, vsl1);
  // CHECK: call i128 @llvm.s390.vmog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmog
  vulll = vec_mulo(vul, vul1);
  // CHECK: call i128 @llvm.s390.vmlog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmlog
}

