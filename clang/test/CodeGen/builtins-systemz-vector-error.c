// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -target-cpu z13 -triple s390x-unknown-unknown \
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

volatile vec_schar vsc;
volatile vec_sshort vss;
volatile vec_sint vsi;
volatile vec_slong vsl;
volatile vec_uchar vuc;
volatile vec_ushort vus;
volatile vec_uint vui;
volatile vec_ulong vul;
volatile vec_double vd;

volatile unsigned int len;
const void * volatile cptr;
int cc;

void test_core(void) {
  __builtin_s390_lcbb(cptr, -1);       // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_lcbb(cptr, 16);       // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_lcbb(cptr, len);      // expected-error {{must be a constant integer}}

  __builtin_s390_vlbb(cptr, -1);       // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vlbb(cptr, 16);       // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vlbb(cptr, len);      // expected-error {{must be a constant integer}}

  __builtin_s390_vpdi(vul, vul, -1);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vpdi(vul, vul, 16);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vpdi(vul, vul, len);  // expected-error {{must be a constant integer}}
}

void test_integer(void) {
  __builtin_s390_verimb(vuc, vuc, vuc, -1);    // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimb(vuc, vuc, vuc, 256);   // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimb(vuc, vuc, vuc, len);   // expected-error {{must be a constant integer}}

  __builtin_s390_verimh(vus, vus, vus, -1);    // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimh(vus, vus, vus, 256);   // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimh(vus, vus, vus, len);   // expected-error {{must be a constant integer}}

  __builtin_s390_verimf(vui, vui, vui, -1);    // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimf(vui, vui, vui, 256);   // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimf(vui, vui, vui, len);   // expected-error {{must be a constant integer}}

  __builtin_s390_verimg(vul, vul, vul, -1);    // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimg(vul, vul, vul, 256);   // expected-error {{argument should be a value from 0 to 255}}
  __builtin_s390_verimg(vul, vul, vul, len);   // expected-error {{must be a constant integer}}

  __builtin_s390_vsldb(vuc, vuc, -1);          // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vsldb(vuc, vuc, 16);          // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vsldb(vuc, vuc, len);         // expected-error {{must be a constant integer}}
}

void test_string(void) {
  __builtin_s390_vfaeb(vuc, vuc, -1);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaeb(vuc, vuc, 16);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaeb(vuc, vuc, len);              // expected-error {{must be a constant integer}}

  __builtin_s390_vfaeh(vus, vus, -1);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaeh(vus, vus, 16);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaeh(vus, vus, len);              // expected-error {{must be a constant integer}}

  __builtin_s390_vfaef(vui, vui, -1);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaef(vui, vui, 16);               // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaef(vui, vui, len);              // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezb(vuc, vuc, -1);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezb(vuc, vuc, 16);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezb(vuc, vuc, len);             // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezh(vus, vus, -1);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezh(vus, vus, 16);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezh(vus, vus, len);             // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezf(vui, vui, -1);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezf(vui, vui, 16);              // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezf(vui, vui, len);             // expected-error {{must be a constant integer}}

  __builtin_s390_vstrcb(vuc, vuc, vuc, -1);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcb(vuc, vuc, vuc, 16);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcb(vuc, vuc, vuc, len);        // expected-error {{must be a constant integer}}

  __builtin_s390_vstrch(vus, vus, vus, -1);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrch(vus, vus, vus, 16);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrch(vus, vus, vus, len);        // expected-error {{must be a constant integer}}

  __builtin_s390_vstrcf(vui, vui, vui, -1);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcf(vui, vui, vui, 16);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcf(vui, vui, vui, len);        // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczb(vuc, vuc, vuc, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczb(vuc, vuc, vuc, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczb(vuc, vuc, vuc, len);       // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczh(vus, vus, vus, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczh(vus, vus, vus, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczh(vus, vus, vus, len);       // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczf(vui, vui, vui, -1);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczf(vui, vui, vui, 16);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczf(vui, vui, vui, len);       // expected-error {{must be a constant integer}}

  __builtin_s390_vfaebs(vuc, vuc, -1, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaebs(vuc, vuc, 16, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaebs(vuc, vuc, len, &cc);        // expected-error {{must be a constant integer}}

  __builtin_s390_vfaehs(vus, vus, -1, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaehs(vus, vus, 16, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaehs(vus, vus, len, &cc);        // expected-error {{must be a constant integer}}

  __builtin_s390_vfaefs(vui, vui, -1, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaefs(vui, vui, 16, &cc);         // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaefs(vui, vui, len, &cc);        // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezbs(vuc, vuc, -1, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezbs(vuc, vuc, 16, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezbs(vuc, vuc, len, &cc);       // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezhs(vus, vus, -1, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezhs(vus, vus, 16, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezhs(vus, vus, len, &cc);       // expected-error {{must be a constant integer}}

  __builtin_s390_vfaezfs(vui, vui, -1, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezfs(vui, vui, 16, &cc);        // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfaezfs(vui, vui, len, &cc);       // expected-error {{must be a constant integer}}

  __builtin_s390_vstrcbs(vuc, vuc, vuc, -1, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcbs(vuc, vuc, vuc, 16, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcbs(vuc, vuc, vuc, len, &cc);  // expected-error {{must be a constant integer}}

  __builtin_s390_vstrchs(vus, vus, vus, -1, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrchs(vus, vus, vus, 16, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrchs(vus, vus, vus, len, &cc);  // expected-error {{must be a constant integer}}

  __builtin_s390_vstrcfs(vui, vui, vui, -1, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcfs(vui, vui, vui, 16, &cc);   // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrcfs(vui, vui, vui, len, &cc);  // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczbs(vuc, vuc, vuc, -1, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczbs(vuc, vuc, vuc, 16, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczbs(vuc, vuc, vuc, len, &cc); // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczhs(vus, vus, vus, -1, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczhs(vus, vus, vus, 16, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczhs(vus, vus, vus, len, &cc); // expected-error {{must be a constant integer}}

  __builtin_s390_vstrczfs(vui, vui, vui, -1, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczfs(vui, vui, vui, 16, &cc);  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vstrczfs(vui, vui, vui, len, &cc); // expected-error {{must be a constant integer}}
}

void test_float(void) {
  __builtin_s390_vftcidb(vd, -1, &cc);              // expected-error {{argument should be a value from 0 to 4095}}
  __builtin_s390_vftcidb(vd, 4096, &cc);            // expected-error {{argument should be a value from 0 to 4095}}
  __builtin_s390_vftcidb(vd, len, &cc);             // expected-error {{must be a constant integer}}

  __builtin_s390_vfidb(vd, -1, 0);                  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfidb(vd, 16, 0);                  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfidb(vd, len, 0);                 // expected-error {{must be a constant integer}}
  __builtin_s390_vfidb(vd, 0, -1);                  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfidb(vd, 0, 16);                  // expected-error {{argument should be a value from 0 to 15}}
  __builtin_s390_vfidb(vd, 0, len);                 // expected-error {{must be a constant integer}}
}
