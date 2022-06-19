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

void test_nnp_assist(void) {
  __builtin_s390_vclfnhs(vus, -1);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vclfnhs(vus, 16);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vclfnhs(vus, len);          // expected-error {{must be a constant integer}}

  __builtin_s390_vclfnls(vus, -1);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vclfnls(vus, 16);           // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vclfnls(vus, len);          // expected-error {{must be a constant integer}}

  __builtin_s390_vcrnfs(vf, vf, -1);         // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcrnfs(vf, vf, 16);         // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcrnfs(vf, vf, len);        // expected-error {{must be a constant integer}}

  __builtin_s390_vcfn(vus, -1);              // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcfn(vus, 16);              // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcfn(vus, len);             // expected-error {{must be a constant integer}}

  __builtin_s390_vcnf(vus, -1);              // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcnf(vus, 16);              // expected-error-re {{argument value {{.*}} is outside the valid range}}
  __builtin_s390_vcnf(vus, len);             // expected-error {{must be a constant integer}}
}

