// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-linux-gnu \
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

void test_integer(void) {
  vsc = vec_sldb(vsc, vsc, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vuc = vec_sldb(vuc, vuc, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vss = vec_sldb(vss, vss, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vus = vec_sldb(vus, vus, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vsi = vec_sldb(vsi, vsi, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vui = vec_sldb(vui, vui, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vsl = vec_sldb(vsl, vsl, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vul = vec_sldb(vul, vul, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vf = vec_sldb(vf, vf, idx);    // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vd = vec_sldb(vd, vd, idx);    // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsld' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}

  vsc = vec_srdb(vsc, vsc, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vuc = vec_srdb(vuc, vuc, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vss = vec_srdb(vss, vss, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vus = vec_srdb(vus, vus, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vsi = vec_srdb(vsi, vsi, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vui = vec_srdb(vui, vui, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vsl = vec_srdb(vsl, vsl, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vul = vec_srdb(vul, vul, idx); // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vf = vec_srdb(vf, vf, idx);    // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
  vd = vec_srdb(vd, vd, idx);    // expected-error {{no matching function}} \
                                 // expected-error {{argument to '__builtin_s390_vsrd' must be a constant integer}}
                                 // expected-note@vecintrin.h:* 9 {{candidate function not viable}}
                                 // expected-note@vecintrin.h:* 1 {{must be a constant integer from 0 to 7}}
}
