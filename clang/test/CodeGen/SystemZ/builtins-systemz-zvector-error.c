// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-linux-gnu \
// RUN: -fzvector -flax-vector-conversions=none \
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
volatile vector double vd;

volatile signed char sc;
volatile signed short ss;
volatile signed int si;
volatile signed long long sl;
volatile unsigned char uc;
volatile unsigned short us;
volatile unsigned int ui;
volatile unsigned long long ul;
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
  len = __lcbb(cptr, idx);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* {{must be a constant power of 2 from 64 to 4096}}
  len = __lcbb(cptr, 200);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* {{must be a constant power of 2 from 64 to 4096}}
  len = __lcbb(cptr, 32);    // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* {{must be a constant power of 2 from 64 to 4096}}
  len = __lcbb(cptr, 8192);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* {{must be a constant power of 2 from 64 to 4096}}

  vsl = vec_permi(vsl, vsl, idx); // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsl = vec_permi(vsl, vsl, -1);  // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsl = vec_permi(vsl, vsl, 4);   // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vul = vec_permi(vul, vul, idx); // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vul = vec_permi(vul, vul, -1);  // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vul = vec_permi(vul, vul, 4);   // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbl = vec_permi(vbl, vbl, idx); // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbl = vec_permi(vbl, vbl, -1);  // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbl = vec_permi(vbl, vbl, 4);   // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 2 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vd = vec_permi(vd, vd, idx);    // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vd = vec_permi(vd, vd, -1);     // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vd = vec_permi(vd, vd, 4);      // expected-error {{no matching function}}
                                  // expected-note@vecintrin.h:* 3 {{candidate function not viable}}
                                  // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}

  vsi = vec_gather_element(vsi, vui, cptrsi, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsi = vec_gather_element(vsi, vui, cptrsi, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsi = vec_gather_element(vsi, vui, cptrsi, 4);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vui = vec_gather_element(vui, vui, cptrui, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vui = vec_gather_element(vui, vui, cptrui, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vui = vec_gather_element(vui, vui, cptrui, 4);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_gather_element(vbi, vui, cptrui, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_gather_element(vbi, vui, cptrui, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_gather_element(vbi, vui, cptrui, 4);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vsl = vec_gather_element(vsl, vul, cptrsl, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vsl = vec_gather_element(vsl, vul, cptrsl, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vsl = vec_gather_element(vsl, vul, cptrsl, 2);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vul = vec_gather_element(vul, vul, cptrul, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vul = vec_gather_element(vul, vul, cptrul, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vul = vec_gather_element(vul, vul, cptrul, 2);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_gather_element(vbl, vul, cptrul, idx); // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_gather_element(vbl, vul, cptrul, -1);  // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_gather_element(vbl, vul, cptrul, 2);   // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vd = vec_gather_element(vd, vul, cptrd, idx);    // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_gather_element(vd, vul, cptrd, -1);     // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_gather_element(vd, vul, cptrd, 2);      // expected-error {{no matching function}}
                                                   // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                                   // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}

  vec_scatter_element(vsi, vui, ptrsi, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vsi, vui, ptrsi, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vsi, vui, ptrsi, 4);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vui, vui, ptrui, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vui, vui, ptrui, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vui, vui, ptrui, 4);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vbi, vui, ptrui, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vbi, vui, ptrui, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vbi, vui, ptrui, 4);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vec_scatter_element(vsl, vul, ptrsl, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vsl, vul, ptrsl, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vsl, vul, ptrsl, 2);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vul, vul, ptrul, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vul, vul, ptrul, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vul, vul, ptrul, 2);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vbl, vul, ptrul, idx); // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vbl, vul, ptrul, -1);  // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vbl, vul, ptrul, 2);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 5 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vd, vul, ptrd, idx);   // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vd, vul, ptrd, -1);    // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vec_scatter_element(vd, vul, ptrd, 2);     // expected-error {{no matching function}}
                                             // expected-note@vecintrin.h:* 6 {{candidate function not viable}}
                                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}

  vsc = vec_load_bndry(cptrsc, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vsc = vec_load_bndry(cptrsc, 200);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vsc = vec_load_bndry(cptrsc, 32);    // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vsc = vec_load_bndry(cptrsc, 8192);  // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vuc = vec_load_bndry(cptruc, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vss = vec_load_bndry(cptrss, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vus = vec_load_bndry(cptrus, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vsi = vec_load_bndry(cptrsi, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vui = vec_load_bndry(cptrui, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vsl = vec_load_bndry(cptrsl, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}
  vul = vec_load_bndry(cptrul, idx);   // expected-error {{no matching function}}
                                       // expected-note@vecintrin.h:* 9 {{must be a constant power of 2 from 64 to 4096}}

  vuc = vec_genmask(idx);  // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* {{must be a constant integer}}

  vuc = vec_genmasks_8(0, idx);    // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vuc = vec_genmasks_8(idx, 0);    // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vuc = vec_genmasks_8(idx, idx);  // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vus = vec_genmasks_16(0, idx);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vus = vec_genmasks_16(idx, 0);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vus = vec_genmasks_16(idx, idx); // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vui = vec_genmasks_32(0, idx);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vui = vec_genmasks_32(idx, 0);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vui = vec_genmasks_32(idx, idx); // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vul = vec_genmasks_64(0, idx);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vul = vec_genmasks_64(idx, 0);   // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}
  vul = vec_genmasks_64(idx, idx); // expected-error {{no matching function}}
                                   // expected-note@vecintrin.h:* {{must be a constant integer}}

  vsc = vec_splat(vsc, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vsc = vec_splat(vsc, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vsc = vec_splat(vsc, 16);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vuc = vec_splat(vuc, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vuc = vec_splat(vuc, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vuc = vec_splat(vuc, 16);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vbc = vec_splat(vbc, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vbc = vec_splat(vbc, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vbc = vec_splat(vbc, 16);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vss = vec_splat(vss, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vss = vec_splat(vss, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vss = vec_splat(vss, 8);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vus = vec_splat(vus, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vus = vec_splat(vus, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vus = vec_splat(vus, 8);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vbs = vec_splat(vbs, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vbs = vec_splat(vbs, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vbs = vec_splat(vbs, 8);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 7}}
  vsi = vec_splat(vsi, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsi = vec_splat(vsi, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsi = vec_splat(vsi, 4);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vui = vec_splat(vui, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vui = vec_splat(vui, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vui = vec_splat(vui, 4);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_splat(vbi, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_splat(vbi, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vbi = vec_splat(vbi, 4);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 3}}
  vsl = vec_splat(vsl, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vsl = vec_splat(vsl, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vsl = vec_splat(vsl, 2);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vul = vec_splat(vul, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vul = vec_splat(vul, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vul = vec_splat(vul, 2);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_splat(vbl, idx); // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_splat(vbl, -1);  // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vbl = vec_splat(vbl, 2);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 1}}
  vd = vec_splat(vd, idx);   // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_splat(vd, -1);    // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}
  vd = vec_splat(vd, 2);     // expected-error {{no matching function}}
                             // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                             // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 1}}

  vsc = vec_splat_s8(idx);  // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vuc = vec_splat_u8(idx);  // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vss = vec_splat_s16(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vus = vec_splat_u16(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vsi = vec_splat_s32(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vui = vec_splat_u32(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vsl = vec_splat_s64(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
  vul = vec_splat_u64(idx); // expected-error {{no matching function}}
                            // expected-note@vecintrin.h:* {{must be a constant integer}}
}

void test_integer(void) {
  vsc = vec_rl_mask(vsc, vuc, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vuc = vec_rl_mask(vuc, vuc, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vss = vec_rl_mask(vss, vus, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vus = vec_rl_mask(vus, vus, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vsi = vec_rl_mask(vsi, vui, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vui = vec_rl_mask(vui, vui, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vsl = vec_rl_mask(vsl, vul, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vul = vec_rl_mask(vul, vul, idx); // expected-error {{no matching function}}
                                    // expected-note@vecintrin.h:* 7 {{candidate function not viable}}
                                    // expected-note@vecintrin.h:* 1 {{must be a constant integer}}

  vsc = vec_sld(vsc, vsc, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vsc = vec_sld(vsc, vsc, -1);  // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vsc = vec_sld(vsc, vsc, 16);  // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vuc = vec_sld(vuc, vuc, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vuc = vec_sld(vuc, vuc, -1);  // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vuc = vec_sld(vuc, vuc, 16);  // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vss = vec_sld(vss, vss, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vus = vec_sld(vus, vus, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vsi = vec_sld(vsi, vsi, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vui = vec_sld(vui, vui, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vsl = vec_sld(vsl, vsl, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}
  vul = vec_sld(vul, vul, idx); // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 11 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 2 {{must be a constant integer from 0 to 15}}
  vd = vec_sld(vd, vd, idx);    // expected-error {{no matching function}}
                                // expected-note@vecintrin.h:* 12 {{candidate function not viable}}
                                // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 15}}

  vsc = vec_sldw(vsc, vsc, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsc = vec_sldw(vsc, vsc, -1);  // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsc = vec_sldw(vsc, vsc, 4);   // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vuc = vec_sldw(vuc, vuc, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vuc = vec_sldw(vuc, vuc, -1);  // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vuc = vec_sldw(vuc, vuc, 4);   // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vss = vec_sldw(vss, vss, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vus = vec_sldw(vus, vus, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsi = vec_sldw(vsi, vsi, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vui = vec_sldw(vui, vui, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vsl = vec_sldw(vsl, vsl, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vul = vec_sldw(vul, vul, idx); // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
  vd = vec_sldw(vd, vd, idx);    // expected-error {{no matching function}}
                                 // expected-note@vecintrin.h:* 8 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 3}}
}

void test_float(void) {
  vd = vec_ctd(vsl, idx); // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vd = vec_ctd(vsl, -1);  // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vd = vec_ctd(vsl, 32);  // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vd = vec_ctd(vul, idx); // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vd = vec_ctd(vul, -1);  // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vd = vec_ctd(vul, 32);  // expected-error {{no matching function}}
                          // expected-note@vecintrin.h:* 1 {{candidate function not viable}}
                          // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}

  vsl = vec_ctsl(vd, idx); // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vsl = vec_ctsl(vd, -1);  // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vsl = vec_ctsl(vd, 32);  // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vul = vec_ctul(vd, idx); // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vul = vec_ctul(vd, -1);  // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}
  vul = vec_ctul(vd, 32);  // expected-error {{no matching function}}
                           // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 31}}

  vbl = vec_fp_test_data_class(vd, idx, &cc);  // expected-error {{must be a constant integer}}
  vbl = vec_fp_test_data_class(vd, -1, &cc);   // expected-error-re {{argument value {{.*}} is outside the valid range}}
  vbl = vec_fp_test_data_class(vd, 4096, &cc); // expected-error-re {{argument value {{.*}} is outside the valid range}}
}
