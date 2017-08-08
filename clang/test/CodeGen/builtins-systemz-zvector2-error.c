// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-linux-gnu \
// RUN: -fzvector -fno-lax-vector-conversions \
// RUN: -Wall -Wno-unused -Werror -fsyntax-only -verify %s

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
  vf = vec_gather_element(vf, vui, cptrf, idx);    // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vf = vec_gather_element(vf, vui, cptrf, -1);     // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vf = vec_gather_element(vf, vui, cptrf, 4);      // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vd = vec_gather_element(vd, vul, cptrd, idx);    // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_gather_element(vd, vul, cptrd, -1);     // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_gather_element(vd, vul, cptrd, 2);      // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}

  vec_scatter_element(vf, vui, ptrf, idx);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vf, vui, ptrf, -1);    // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vf, vui, ptrf, 4);     // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vd, vul, ptrd, idx);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vd, vul, ptrd, -1);    // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vd, vul, ptrd, 2);     // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}

  vf = vec_splat(vf, idx);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vf = vec_splat(vf, -1);    // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vf = vec_splat(vf, 4);     // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vd = vec_splat(vd, idx);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_splat(vd, -1);    // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_splat(vd, 2);     // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
}

void test_integer(void) {
  vf = vec_sld(vf, vf, idx);    // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vd = vec_sld(vd, vd, idx);    // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}

  vuc = vec_msum_u128(vul, vul, vuc, idx);  // expected-error {{must be a constant integer}}
  vuc = vec_msum_u128(vul, vul, vuc, -1);   // expected-error {{should be a value from 0 to 15}}
  vuc = vec_msum_u128(vul, vul, vuc, 16);   // expected-error {{should be a value from 0 to 15}}
}

void test_float(void) {
  vbi = vec_fp_test_data_class(vf, idx, &cc);   // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
  vbi = vec_fp_test_data_class(vf, -1, &cc);    // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
  vbi = vec_fp_test_data_class(vf, 4096, &cc);  // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
  vbl = vec_fp_test_data_class(vd, idx, &cc);   // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
  vbl = vec_fp_test_data_class(vd, -1, &cc);    // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
  vbl = vec_fp_test_data_class(vd, 4096, &cc);  // expected-error {{no matching function}}
                                                // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 4095}}
}
