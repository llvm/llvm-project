// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-linux-gnu \
// RUN: -O -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-linux-gnu \
// RUN: -O -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -S %s -o - | FileCheck %s --check-prefix=CHECK-ASM

#include <vecintrin.h>

volatile vector signed char vsc;
volatile vector signed short vss;
volatile vector signed int vsi;
volatile vector signed long long vsl;
volatile vector unsigned char vuc;
volatile vector unsigned short vus;
volatile vector unsigned int vui;
volatile vector unsigned long long vul;
volatile vector bool char vbc;
volatile vector bool short vbs;
volatile vector bool int vbi;
volatile vector bool long long vbl;
volatile vector float vf;
volatile vector double vd;

volatile signed char sc;
volatile signed short ss;
volatile signed int si;
volatile signed long long sl;
volatile unsigned char uc;
volatile unsigned short us;
volatile unsigned int ui;
volatile unsigned long long ul;
volatile float f;
volatile double d;

const void * volatile cptr;
const signed char * volatile cptrsc;
const signed short * volatile cptrss;
const signed int * volatile cptrsi;
const signed long long * volatile cptrsl;
const unsigned char * volatile cptruc;
const unsigned short * volatile cptrus;
const unsigned int * volatile cptrui;
const unsigned long long * volatile cptrul;
const float * volatile cptrf;
const double * volatile cptrd;

void * volatile ptr;
signed char * volatile ptrsc;
signed short * volatile ptrss;
signed int * volatile ptrsi;
signed long long * volatile ptrsl;
unsigned char * volatile ptruc;
unsigned short * volatile ptrus;
unsigned int * volatile ptrui;
unsigned long long * volatile ptrul;
float * volatile ptrf;
double * volatile ptrd;

volatile unsigned int len;
volatile int idx;
int cc;

void test_core(void) {
  // CHECK-ASM-LABEL: test_core
  vector signed short vss2;
  vector signed int vsi2;
  vector signed long long vsl2;
  vector unsigned short vus2;
  vector unsigned int vui2;
  vector unsigned long long vul2;
  vector float vf2;
  vector double vd2;

  vss += vec_revb(vec_xl(idx, cptrss));
  // CHECK-ASM: vlbrh
  vus += vec_revb(vec_xl(idx, cptrus));
  // CHECK-ASM: vlbrh
  vsi += vec_revb(vec_xl(idx, cptrsi));
  // CHECK-ASM: vlbrf
  vui += vec_revb(vec_xl(idx, cptrui));
  // CHECK-ASM: vlbrf
  vsl += vec_revb(vec_xl(idx, cptrsl));
  // CHECK-ASM: vlbrg
  vul += vec_revb(vec_xl(idx, cptrul));
  // CHECK-ASM: vlbrg
  vf += vec_revb(vec_xl(idx, cptrf));
  // CHECK-ASM: vlbrf
  vd += vec_revb(vec_xl(idx, cptrd));
  // CHECK-ASM: vlbrg

  vec_xst(vec_revb(vss), idx, ptrss);
  // CHECK-ASM: vstbrh
  vec_xst(vec_revb(vus), idx, ptrus);
  // CHECK-ASM: vstbrh
  vec_xst(vec_revb(vsi), idx, ptrsi);
  // CHECK-ASM: vstbrf
  vec_xst(vec_revb(vui), idx, ptrui);
  // CHECK-ASM: vstbrf
  vec_xst(vec_revb(vsl), idx, ptrsl);
  // CHECK-ASM: vstbrg
  vec_xst(vec_revb(vul), idx, ptrul);
  // CHECK-ASM: vstbrg
  vec_xst(vec_revb(vf), idx, ptrf);
  // CHECK-ASM: vstbrf
  vec_xst(vec_revb(vd), idx, ptrd);
  // CHECK-ASM: vstbrg

  vss += vec_revb(vec_insert_and_zero(cptrss));
  // CHECK-ASM: vllebrzh
  vus += vec_revb(vec_insert_and_zero(cptrus));
  // CHECK-ASM: vllebrzh
  vsi += vec_revb(vec_insert_and_zero(cptrsi));
  // CHECK-ASM: vllebrzf
  vui += vec_revb(vec_insert_and_zero(cptrui));
  // CHECK-ASM: vllebrzf
  vsl += vec_revb(vec_insert_and_zero(cptrsl));
  // CHECK-ASM: vllebrzg
  vul += vec_revb(vec_insert_and_zero(cptrul));
  // CHECK-ASM: vllebrzg
  vf += vec_revb(vec_insert_and_zero(cptrf));
  // CHECK-ASM: vllebrzf
  vd += vec_revb(vec_insert_and_zero(cptrd));
  // CHECK-ASM: vllebrzg

  vss += vec_revb(vec_splats(ss));
  // CHECK-ASM: vlbrreph
  vus += vec_revb(vec_splats(us));
  // CHECK-ASM: vlbrreph
  vsi += vec_revb(vec_splats(si));
  // CHECK-ASM: vlbrrepf
  vui += vec_revb(vec_splats(ui));
  // CHECK-ASM: vlbrrepf
  vsl += vec_revb(vec_splats(sl));
  // CHECK-ASM: vlbrrepg
  vul += vec_revb(vec_splats(ul));
  // CHECK-ASM: vlbrrepg
  vf += vec_revb(vec_splats(f));
  // CHECK-ASM: vlbrrepf
  vd += vec_revb(vec_splats(d));
  // CHECK-ASM: vlbrrepg

  vus = vec_splats(__builtin_bswap16(us));
  // CHECK-ASM: vlbrreph
  vui = vec_splats(__builtin_bswap32(ui));
  // CHECK-ASM: vlbrrepf
  vul = vec_splats((unsigned long long)__builtin_bswap64(ul));
  // CHECK-ASM: vlbrrepg

  vss2 = vss;
  vss += vec_revb(vec_insert(ss, vec_revb(vss2), 0));
  // CHECK-ASM: vlebrh
  vus2 = vus;
  vus += vec_revb(vec_insert(us, vec_revb(vus2), 0));
  // CHECK-ASM: vlebrh
  vsi2 = vsi;
  vsi += vec_revb(vec_insert(si, vec_revb(vsi2), 0));
  // CHECK-ASM: vlebrf
  vui2 = vui;
  vui += vec_revb(vec_insert(ui, vec_revb(vui2), 0));
  // CHECK-ASM: vlebrf
  vsl2 = vsl;
  vsl += vec_revb(vec_insert(sl, vec_revb(vsl2), 0));
  // CHECK-ASM: vlebrg
  vul2 = vul;
  vul += vec_revb(vec_insert(ul, vec_revb(vul2), 0));
  // CHECK-ASM: vlebrg
  vf2 = vf;
  vf += vec_revb(vec_insert(f, vec_revb(vf2), 0));
  // CHECK-ASM: vlebrf
  vd2 = vd;
  vd += vec_revb(vec_insert(d, vec_revb(vd2), 0));
  // CHECK-ASM: vlebrg

  vus2 = vus;
  vus = vec_insert(__builtin_bswap16(us), vus2, 0);
  // CHECK-ASM: vlebrh
  vui2 = vui;
  vui = vec_insert(__builtin_bswap32(ui), vui2, 0);
  // CHECK-ASM: vlebrf
  vul2 = vul;
  vul = vec_insert(__builtin_bswap64(ul), vul2, 0);
  // CHECK-ASM: vlebrg

  ss = vec_extract(vec_revb(vss), 0);
  // CHECK-ASM: vstebrh
  us = vec_extract(vec_revb(vus), 0);
  // CHECK-ASM: vstebrh
  si = vec_extract(vec_revb(vsi), 0);
  // CHECK-ASM: vstebrf
  ui = vec_extract(vec_revb(vui), 0);
  // CHECK-ASM: vstebrf
  sl = vec_extract(vec_revb(vsl), 0);
  // CHECK-ASM: vstebrg
  ul = vec_extract(vec_revb(vul), 0);
  // CHECK-ASM: vstebrg
  f = vec_extract(vec_revb(vf), 0);
  // CHECK-ASM: vstebrf
  d = vec_extract(vec_revb(vd), 0);
  // CHECK-ASM: vstebrg

  us = __builtin_bswap16(vec_extract(vus, 0));
  // CHECK-ASM: vstebrh
  ui = __builtin_bswap32(vec_extract(vui, 0));
  // CHECK-ASM: vstebrf
  ul = __builtin_bswap64(vec_extract(vul, 0));
  // CHECK-ASM: vstebrg

  vsc += vec_reve(vec_xl(idx, cptrsc));
  // CHECK-ASM: vlbrq
  vuc += vec_reve(vec_xl(idx, cptruc));
  // CHECK-ASM: vlbrq
  vss += vec_reve(vec_xl(idx, cptrss));
  // CHECK-ASM: vlerh
  vus += vec_reve(vec_xl(idx, cptrus));
  // CHECK-ASM: vlerh
  vsi += vec_reve(vec_xl(idx, cptrsi));
  // CHECK-ASM: vlerf
  vui += vec_reve(vec_xl(idx, cptrui));
  // CHECK-ASM: vlerf
  vsl += vec_reve(vec_xl(idx, cptrsl));
  // CHECK-ASM: vlerg
  vul += vec_reve(vec_xl(idx, cptrul));
  // CHECK-ASM: vlerg
  vf += vec_reve(vec_xl(idx, cptrf));
  // CHECK-ASM: vlerf
  vd += vec_reve(vec_xl(idx, cptrd));
  // CHECK-ASM: vlerg

  vec_xst(vec_reve(vsc), idx, ptrsc);
  // CHECK-ASM: vstbrq
  vec_xst(vec_reve(vuc), idx, ptruc);
  // CHECK-ASM: vstbrq
  vec_xst(vec_reve(vss), idx, ptrss);
  // CHECK-ASM: vsterh
  vec_xst(vec_reve(vus), idx, ptrus);
  // CHECK-ASM: vsterh
  vec_xst(vec_reve(vsi), idx, ptrsi);
  // CHECK-ASM: vsterf
  vec_xst(vec_reve(vui), idx, ptrui);
  // CHECK-ASM: vsterf
  vec_xst(vec_reve(vsl), idx, ptrsl);
  // CHECK-ASM: vsterg
  vec_xst(vec_reve(vul), idx, ptrul);
  // CHECK-ASM: vsterg
  vec_xst(vec_reve(vf), idx, ptrf);
  // CHECK-ASM: vsterf
  vec_xst(vec_reve(vd), idx, ptrd);
  // CHECK-ASM: vsterg
}

void test_integer(void) {
  // CHECK-ASM-LABEL: test_integer

  vsc = vec_sldb(vsc, vsc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vsc = vec_sldb(vsc, vsc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vuc = vec_sldb(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vuc = vec_sldb(vuc, vuc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vss = vec_sldb(vss, vss, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vss = vec_sldb(vss, vss, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vus = vec_sldb(vus, vus, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vus = vec_sldb(vus, vus, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vsi = vec_sldb(vsi, vsi, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vsi = vec_sldb(vsi, vsi, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vui = vec_sldb(vui, vui, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vui = vec_sldb(vui, vui, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vsl = vec_sldb(vsl, vsl, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vsl = vec_sldb(vsl, vsl, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vul = vec_sldb(vul, vul, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vul = vec_sldb(vul, vul, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vf = vec_sldb(vf, vf, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vf = vec_sldb(vf, vf, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld
  vd = vec_sldb(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsld
  vd = vec_sldb(vd, vd, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsld(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsld

  vsc = vec_srdb(vsc, vsc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vsc = vec_srdb(vsc, vsc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vuc = vec_srdb(vuc, vuc, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vuc = vec_srdb(vuc, vuc, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vss = vec_srdb(vss, vss, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vss = vec_srdb(vss, vss, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vus = vec_srdb(vus, vus, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vus = vec_srdb(vus, vus, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vsi = vec_srdb(vsi, vsi, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vsi = vec_srdb(vsi, vsi, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vui = vec_srdb(vui, vui, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vui = vec_srdb(vui, vui, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vsl = vec_srdb(vsl, vsl, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vsl = vec_srdb(vsl, vsl, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vul = vec_srdb(vul, vul, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vul = vec_srdb(vul, vul, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vf = vec_srdb(vf, vf, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vf = vec_srdb(vf, vf, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
  vd = vec_srdb(vd, vd, 0);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 0)
  // CHECK-ASM: vsrd
  vd = vec_srdb(vd, vd, 7);
  // CHECK: call <16 x i8> @llvm.s390.vsrd(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i32 7)
  // CHECK-ASM: vsrd
}

void test_string(void) {
  // CHECK-ASM-LABEL: test_string

  vuc = vec_search_string_cc(vsc, vsc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vbc, vbc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vss, vss, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vbs, vbs, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vus, vus, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vsi, vsi, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vbi, vbi, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0
  vuc = vec_search_string_cc(vui, vui, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrsf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrsf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 0

  vuc = vec_search_string_until_zero_cc(vsc, vsc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vbc, vbc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vuc, vuc, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszb(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszb %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vss, vss, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vbs, vbs, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vus, vus, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszh(<8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszh %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vsi, vsi, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vbi, vbi, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
  vuc = vec_search_string_until_zero_cc(vui, vui, vuc, &cc);
  // CHECK: call { <16 x i8>, i32 } @llvm.s390.vstrszf(<4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <16 x i8> %{{.*}})
  // CHECK-ASM: vstrszf %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
}

void test_float(void) {
  // CHECK-ASM-LABEL: test_float

  vd = vec_double(vsl);
  // CHECK: sitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK-ASM: vcdgb
  vd = vec_double(vul);
  // CHECK: uitofp <2 x i64> %{{.*}} to <2 x double>
  // CHECK-ASM: vcdlgb
  vf = vec_float(vsi);
  // CHECK: sitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK-ASM: vcefb
  vf = vec_float(vui);
  // CHECK: uitofp <4 x i32> %{{.*}} to <4 x float>
  // CHECK-ASM: vcelfb

  vsl = vec_signed(vd);
  // CHECK: fptosi <2 x double> %{{.*}} to <2 x i64>
  // CHECK-ASM: vcgdb
  vsi = vec_signed(vf);
  // CHECK: fptosi <4 x float> %{{.*}} to <4 x i32>
  // CHECK-ASM: vcfeb
  vul = vec_unsigned(vd);
  // CHECK: fptoui <2 x double> %{{.*}} to <2 x i64>
  // CHECK-ASM: vclgdb
  vui = vec_unsigned(vf);
  // CHECK: fptoui <4 x float> %{{.*}} to <4 x i32>
  // CHECK-ASM: vclfeb
}

