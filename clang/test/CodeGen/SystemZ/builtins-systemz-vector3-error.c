// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z15 -triple s390x-unknown-unknown \
// RUN: -Wall -Wno-unused -Werror -fsyntax-only -verify %s

typedef __attribute__((vector_size(16))) signed char vec_schar;
typedef __attribute__((vector_size(16))) signed short vec_sshort;
typedef __attribute__((vector_size(16))) signed int vec_sint;
typedef __attribute__((vector_size(16))) signed long long vec_slong;
typedef __attribute__((vector_size(16))) unsigned char vec_uchar;
typedef __attribute__((vector_size(16))) unsigned short vec_ushort;
typedef __attribute__((vector_size(16))) unsigned int vec_uint;
typedef __attribute__((vector_size(16))) unsigned long long vec_ulong;
typedef __attribute__((vector_size(16))) double vec_double;
typedef __attribute__((vector_size(16))) float vec_float;

volatile vec_schar vsc;
volatile vec_sshort vss;
volatile vec_sint vsi;
volatile vec_slong vsl;
volatile vec_uchar vuc;
volatile vec_ushort vus;
volatile vec_uint vui;
volatile vec_ulong vul;
volatile vec_double vd;
volatile vec_float vf;

volatile unsigned int len;
int cc;

void test_integer(void) {
  __builtin_s390_vsld(vuc, vuc, -1);          // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vsld(vuc, vuc, 8);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vsld(vuc, vuc, len);         // expected-error {{must be a constant integer}}

  __builtin_s390_vsrd(vuc, vuc, -1);          // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vsrd(vuc, vuc, 8);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vsrd(vuc, vuc, len);         // expected-error {{must be a constant integer}}
}

