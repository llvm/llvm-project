// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu future %s -verify
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu future %s -verify

// The use of PPC MMA types is strongly restricted. Non-pointer MMA variables
// can only be declared in functions and a limited number of operations are
// supported on these types. This test case checks that invalid uses of MMA
// types are correctly prevented.

// vector dmr

// typedef
typedef __dmr1024 dmr_t;

// function argument
void testDmrArg1(__dmr1024 vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  *vdmrp = vdmr;
}

void testDmrArg2(const __dmr1024 vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  *vdmrp = vdmr;
}

void testDmrArg3(const dmr_t vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  *vdmrp = vdmr;
}

// function return
__dmr1024 testDmrRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  return *vdmrp; // expected-error {{invalid use of PPC MMA type}}
}

const dmr_t testDmrRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  return *vdmrp; // expected-error {{invalid use of PPC MMA type}}
}

// global
__dmr1024 globalvdmr;        // expected-error {{invalid use of PPC MMA type}}
const __dmr1024 globalvdmr2; // expected-error {{invalid use of PPC MMA type}}
__dmr1024 *globalvdmrp;
const __dmr1024 *const globalvdmrp2;
dmr_t globalvdmr_t; // expected-error {{invalid use of PPC MMA type}}

// struct field
struct TestDmrStruct {
  int a;
  float b;
  __dmr1024 c; // expected-error {{invalid use of PPC MMA type}}
  __dmr1024 *vq;
};

// operators
int testDmrOperators1(int *ptr) {
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  __dmr1024 vdmr1 = *(vdmrp + 0);
  __dmr1024 vdmr2 = *(vdmrp + 1);
  __dmr1024 vdmr3 = *(vdmrp + 2);
  if (vdmr1) // expected-error {{statement requires expression of scalar type ('__dmr1024' invalid)}}
    *(vdmrp + 10) = vdmr1;
  if (!vdmr2) // expected-error {{invalid argument type '__dmr1024' to unary expression}}
    *(vdmrp + 11) = vdmr3;
  int c1 = vdmr1 && vdmr2; // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  int c2 = vdmr2 == vdmr3; // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  int c3 = vdmr2 < vdmr1;  // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  return c1 || c2 || c3;
}

void testDmrOperators2(int *ptr) {
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  __dmr1024 vdmr1 = *(vdmrp + 0);
  __dmr1024 vdmr2 = *(vdmrp + 1);
  __dmr1024 vdmr3 = *(vdmrp + 2);
  vdmr1 = -vdmr1;        // expected-error {{invalid argument type '__dmr1024' to unary expression}}
  vdmr2 = vdmr1 + vdmr3; // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  vdmr2 = vdmr2 * vdmr3; // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  vdmr3 = vdmr3 | vdmr3; // expected-error {{invalid operands to binary expression ('__dmr1024' and '__dmr1024')}}
  vdmr3 = vdmr3 << 2;    // expected-error {{invalid operands to binary expression ('__dmr1024' and 'int')}}
  *(vdmrp + 10) = vdmr1;
  *(vdmrp + 11) = vdmr2;
  *(vdmrp + 12) = vdmr3;
}

vector unsigned char testDmrOperators3(int *ptr) {
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  __dmr1024 vdmr1 = *(vdmrp + 0);
  __dmr1024 vdmr2 = *(vdmrp + 1);
  __dmr1024 vdmr3 = *(vdmrp + 2);
  vdmr1 ? *(vdmrp + 10) = vdmr2 : *(vdmrp + 11) = vdmr3; // expected-error {{used type '__dmr1024' where arithmetic or pointer type is required}}
  vdmr2 = vdmr3;
  return vdmr2[1]; // expected-error {{subscripted value is not an array, pointer, or vector}}
}

void testDmrOperators4(int v, void *ptr) {
  __dmr1024 *vdmrp = (__dmr1024 *)ptr;
  __dmr1024 vdmr1 = (__dmr1024)v;   // expected-error {{used type '__dmr1024' where arithmetic or pointer type is required}}
  __dmr1024 vdmr2 = (__dmr1024)vdmrp; // expected-error {{used type '__dmr1024' where arithmetic or pointer type is required}}
}
