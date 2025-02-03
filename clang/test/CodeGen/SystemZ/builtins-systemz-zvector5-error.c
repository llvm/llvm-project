// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu arch15 -triple s390x-linux-gnu \
// RUN: -fzvector -flax-vector-conversions=none \
// RUN: -Wall -Wno-unused -Werror -fsyntax-only -verify %s

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

void test_integer(void) {
  vsc = vec_evaluate(vsc, vsc, vsc, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 14 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vuc = vec_evaluate(vuc, vuc, vuc, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vbc = vec_evaluate(vbc, vbc, vbc, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vss = vec_evaluate(vss, vss, vss, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 14 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vus = vec_evaluate(vus, vus, vus, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vbs = vec_evaluate(vbs, vbs, vbs, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vsi = vec_evaluate(vsi, vsi, vsi, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 14 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vui = vec_evaluate(vui, vui, vui, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vbi = vec_evaluate(vbi, vbi, vbi, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vsl = vec_evaluate(vsl, vsl, vsl, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 14 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vul = vec_evaluate(vul, vul, vul, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vbl = vec_evaluate(vbl, vbl, vbl, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vslll = vec_evaluate(vslll, vslll, vslll, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 14 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 1 {{must be a constant integer}}
  vulll = vec_evaluate(vulll, vulll, vulll, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
  vblll = vec_evaluate(vblll, vblll, vblll, idx); // expected-error {{no matching function}} \
                                          // expected-error {{argument to '__builtin_s390_veval' must be a constant integer}} \
                                          // expected-note@vecintrin.h:* 13 {{candidate function not viable}}
                                          // expected-note@vecintrin.h:* 2 {{must be a constant integer}}
}
