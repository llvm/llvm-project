// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z14 -triple s390x-unknown-unknown \
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
  __builtin_s390_vmslg(vul, vul, vuc, -1);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vmslg(vul, vul, vuc, 16);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vmslg(vul, vul, vuc, len);  // expected-error {{must be a constant integer}}
}

void test_float(void) {
  __builtin_s390_vfmaxdb(vd, vd, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmaxdb(vd, vd, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmaxdb(vd, vd, len);       // expected-error {{must be a constant integer}}
  __builtin_s390_vfmindb(vd, vd, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmindb(vd, vd, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmindb(vd, vd, len);       // expected-error {{must be a constant integer}}

  __builtin_s390_vftcisb(vf, -1, &cc);       // expected-error {{argument should be a value from 0 to 4095}}
  __builtin_s390_vftcisb(vf, 4096, &cc);     // expected-error {{argument should be a value from 0 to 4095}}
  __builtin_s390_vftcisb(vf, len, &cc);      // expected-error {{must be a constant integer}}

  __builtin_s390_vfisb(vf, -1, 0);           // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfisb(vf, 16, 0);           // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfisb(vf, len, 0);          // expected-error {{must be a constant integer}}
  __builtin_s390_vfisb(vf, 0, -1);           // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfisb(vf, 0, 16);           // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfisb(vf, 0, len);          // expected-error {{must be a constant integer}}

  __builtin_s390_vfmaxsb(vf, vf, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmaxsb(vf, vf, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfmaxsb(vf, vf, len);       // expected-error {{must be a constant integer}}
  __builtin_s390_vfminsb(vf, vf, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfminsb(vf, vf, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfminsb(vf, vf, len);       // expected-error {{must be a constant integer}}
}
