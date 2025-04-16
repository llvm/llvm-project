// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-linux-gnu \
// RUN: -O2 -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed short vss;
volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector signed __int128 vslll;
volatile vector unsigned char vuc;
volatile vector unsigned short vus;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector unsigned __int128 vulll;
volatile vector bool char vbc;
volatile vector bool short vbs;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector bool __int128 vblll;
volatile vector float vf;
volatile vector double vd;

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

  vsc = vec_blend(vsc, vsc, vsc);
  // CHECK-ASM: vblendb
  vbc = vec_blend(vbc, vbc, vsc);
  // CHECK-ASM: vblendb
  vuc = vec_blend(vuc, vuc, vsc);
  // CHECK-ASM: vblendb
  vss = vec_blend(vss, vss, vss);
  // CHECK-ASM: vblendh
  vbs = vec_blend(vbs, vbs, vss);
  // CHECK-ASM: vblendh
  vus = vec_blend(vus, vus, vss);
  // CHECK-ASM: vblendh
  vsi = vec_blend(vsi, vsi, vsi);
  // CHECK-ASM: vblendf
  vbi = vec_blend(vbi, vbi, vsi);
  // CHECK-ASM: vblendf
  vui = vec_blend(vui, vui, vsi);
  // CHECK-ASM: vblendf
  vsl = vec_blend(vsl, vsl, vsl);
  // CHECK-ASM: vblendg
  vul = vec_blend(vul, vul, vsl);
  // CHECK-ASM: vblendg
  vbl = vec_blend(vbl, vbl, vsl);
  // CHECK-ASM: vblendg
  vslll = vec_blend(vslll, vslll, vslll);
  // CHECK-ASM: vblendq
  vblll = vec_blend(vblll, vblll, vslll);
  // CHECK-ASM: vblendq
  vulll = vec_blend(vulll, vulll, vslll);
  // CHECK-ASM: vblendq
  vf = vec_blend(vf, vf, vsi);
  // CHECK-ASM: vblendf
  vd = vec_blend(vd, vd, vsl);
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

  vblll = vec_cmpeq(vslll, vslll);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq
  vblll = vec_cmpeq(vulll, vulll);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq
  vblll = vec_cmpeq(vblll, vblll);
  // CHECK: icmp eq <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vceqq

  vblll = vec_cmpge(vslll, vslll);
  // CHECK: icmp sge <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmpge(vulll, vulll);
  // CHECK: icmp uge <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmpgt(vslll, vslll);
  // CHECK: icmp sgt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmpgt(vulll, vulll);
  // CHECK: icmp ugt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmple(vslll, vslll);
  // CHECK: icmp sle <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmple(vulll, vulll);
  // CHECK: icmp ule <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  vblll = vec_cmplt(vslll, vslll);
  // CHECK: icmp slt <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchq
  vblll = vec_cmplt(vulll, vulll);
  // CHECK: icmp ult <1 x i128> %{{.*}}, %{{.*}}
  // CHECK-ASM: vchlq

  idx = vec_all_eq(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_eq(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_eq(vblll, vblll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_all_ne(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_ne(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_all_ne(vblll, vblll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_all_ge(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_ge(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_gt(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_gt(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_le(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_le(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_all_lt(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_all_lt(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_eq(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_eq(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_eq(vblll, vblll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_any_ne(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_ne(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs
  idx = vec_any_ne(vblll, vblll);
  // CHECK: call { i128, i32 } @llvm.s390.vceqqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vceqqs

  idx = vec_any_ge(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_ge(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_gt(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_gt(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_le(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_le(vulll, vulll);
  // CHECK: call { i128, i32 } @llvm.s390.vchlqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchlqs

  idx = vec_any_lt(vslll, vslll);
  // CHECK: call { i128, i32 } @llvm.s390.vchqs(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vchqs
  idx = vec_any_lt(vulll, vulll);
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
  // CHECK-ASM: vlcq

  vslll = vec_avg(vslll, vslll);
  // CHECK: call i128 @llvm.s390.vavgq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vavgq
  vulll = vec_avg(vulll, vulll);
  // CHECK: call i128 @llvm.s390.vavglq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vavglq

  vsc = vec_evaluate(vsc, vsc, vsc, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsc = vec_evaluate(vsc, vsc, vsc, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vuc = vec_evaluate(vuc, vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vuc = vec_evaluate(vuc, vuc, vuc, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbc = vec_evaluate(vbc, vbc, vbc, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbc = vec_evaluate(vbc, vbc, vbc, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vss = vec_evaluate(vss, vss, vss, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vss = vec_evaluate(vss, vss, vss, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vus = vec_evaluate(vus, vus, vus, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vus = vec_evaluate(vus, vus, vus, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbs = vec_evaluate(vbs, vbs, vbs, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbs = vec_evaluate(vbs, vbs, vbs, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vsi = vec_evaluate(vsi, vsi, vsi, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsi = vec_evaluate(vsi, vsi, vsi, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vui = vec_evaluate(vui, vui, vui, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vui = vec_evaluate(vui, vui, vui, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbi = vec_evaluate(vbi, vbi, vbi, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbi = vec_evaluate(vbi, vbi, vbi, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vsl = vec_evaluate(vsl, vsl, vsl, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vsl = vec_evaluate(vsl, vsl, vsl, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vul = vec_evaluate(vul, vul, vul, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vul = vec_evaluate(vul, vul, vul, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vbl = vec_evaluate(vbl, vbl, vbl, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vbl = vec_evaluate(vbl, vbl, vbl, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vslll = vec_evaluate(vslll, vslll, vslll, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vslll = vec_evaluate(vslll, vslll, vslll, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vulll = vec_evaluate(vulll, vulll, vulll, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vulll = vec_evaluate(vulll, vulll, vulll, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval
  vblll = vec_evaluate(vblll, vblll, vblll, 0);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: veval
  vblll = vec_evaluate(vblll, vblll, vblll, 255);
  // CHECK: call <16 x i8> @llvm.s390.veval(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 255)
  // CHECK-ASM: veval

  vslll = vec_max(vslll, vslll);
  // CHECK-ASM: vmxq
  vulll = vec_max(vulll, vulll);
  // CHECK-ASM: vmxlq
  vslll = vec_min(vslll, vslll);
  // CHECK-ASM: vmnq
  vulll = vec_min(vulll, vulll);
  // CHECK-ASM: vmnlq

  vsl = vec_mladd(vsl, vsl, vsl);
  // CHECK-ASM: vmalg
  vsl = vec_mladd(vul, vsl, vsl);
  // CHECK-ASM: vmalg
  vsl = vec_mladd(vsl, vul, vul);
  // CHECK-ASM: vmalg
  vul = vec_mladd(vul, vul, vul);
  // CHECK-ASM: vmalg
  vslll = vec_mladd(vslll, vslll, vslll);
  // CHECK-ASM: vmalq
  vslll = vec_mladd(vulll, vslll, vslll);
  // CHECK-ASM: vmalq
  vslll = vec_mladd(vslll, vulll, vulll);
  // CHECK-ASM: vmalq
  vulll = vec_mladd(vulll, vulll, vulll);
  // CHECK-ASM: vmalq

  vsl = vec_mhadd(vsl, vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmahg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmahg
  vul = vec_mhadd(vul, vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmalhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmalhg
  vslll = vec_mhadd(vslll, vslll, vslll);
  // CHECK: call i128 @llvm.s390.vmahq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmahq
  vulll = vec_mhadd(vulll, vulll, vulll);
  // CHECK: call i128 @llvm.s390.vmalhq(i128 %{{.*}}, i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmalhq

  vslll = vec_meadd(vsl, vsl, vslll);
  // CHECK: call i128 @llvm.s390.vmaeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaeg
  vulll = vec_meadd(vul, vul, vulll);
  // CHECK: call i128 @llvm.s390.vmaleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaleg

  vslll = vec_moadd(vsl, vsl, vslll);
  // CHECK: call i128 @llvm.s390.vmaog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmaog
  vulll = vec_moadd(vul, vul, vulll);
  // CHECK: call i128 @llvm.s390.vmalog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmalog

  vsl = vec_mulh(vsl, vsl);
  // CHECK: call <2 x i64> @llvm.s390.vmhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmhg
  vul = vec_mulh(vul, vul);
  // CHECK: call <2 x i64> @llvm.s390.vmlhg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmlhg
  vslll = vec_mulh(vslll, vslll);
  // CHECK: call i128 @llvm.s390.vmhq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmhq
  vulll = vec_mulh(vulll, vulll);
  // CHECK: call i128 @llvm.s390.vmlhq(i128 %{{.*}}, i128 %{{.*}})
  // CHECK-ASM: vmlhq

  vslll = vec_mule(vsl, vsl);
  // CHECK: call i128 @llvm.s390.vmeg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmeg
  vulll = vec_mule(vul, vul);
  // CHECK: call i128 @llvm.s390.vmleg(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmleg

  vslll = vec_mulo(vsl, vsl);
  // CHECK: call i128 @llvm.s390.vmog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmog
  vulll = vec_mulo(vul, vul);
  // CHECK: call i128 @llvm.s390.vmlog(<2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  // CHECK-ASM: vmlog
}

