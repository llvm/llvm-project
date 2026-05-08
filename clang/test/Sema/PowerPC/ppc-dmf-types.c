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
typedef __dmr2048 dmrp_t;

// function argument
void testDmrArg1(dmr_t vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vdmrp = (dmr_t *)ptr;
  *vdmrp = vdmr;
}

void testDmrArg2(const dmr_t vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vdmrp = (dmr_t *)ptr;
  *vdmrp = vdmr;
}

void testDmrArg3(const dmr_t vdmr, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vdmrp = (dmr_t *)ptr;
  *vdmrp = vdmr;
}

void testDmrPArg1(const dmrp_t vdmrp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vdmrpp = (dmrp_t *)ptr;
  *vdmrpp = vdmrp;
}

void testDmrPArg2(const dmrp_t vdmrp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vdmrpp = (dmrp_t *)ptr;
  *vdmrpp = vdmrp;
}

void testDmrPArg3(const dmrp_t vdmrp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vdmrpp = (dmrp_t *)ptr;
  *vdmrpp = vdmrp;
}

// function return
dmr_t testDmrRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vdmrp = (dmr_t *)ptr;
  return *vdmrp; // expected-error {{invalid use of PPC MMA type}}
}

const dmr_t testDmrRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vdmrp = (dmr_t *)ptr;
  return *vdmrp; // expected-error {{invalid use of PPC MMA type}}
}

dmrp_t testDmrPRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vdmrpp = (dmrp_t *)ptr;
  return *vdmrpp; // expected-error {{invalid use of PPC MMA type}}
}

const dmrp_t testDmrPRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vdmrpp = (dmrp_t *)ptr;
  return *vdmrpp; // expected-error {{invalid use of PPC MMA type}}
}

// global
dmr_t globalvdmr;        // expected-error {{invalid use of PPC MMA type}}
const dmr_t globalvdmr2; // expected-error {{invalid use of PPC MMA type}}
dmr_t *globalvdmrp;
const dmr_t *const globalvdmrp2;
dmr_t globalvdmr_t; // expected-error {{invalid use of PPC MMA type}}

dmrp_t globalvdmrp;        // expected-error {{invalid use of PPC MMA type}}
const dmrp_t globalvdmrp2; // expected-error {{invalid use of PPC MMA type}}
dmrp_t *globalvdmrpp;
const dmrp_t *const globalvdmrpp2;
dmrp_t globalvdmrp_t; // expected-error {{invalid use of PPC MMA type}}

// struct field
struct TestDmrStruct {
  int a;
  float b;
  dmr_t c; // expected-error {{invalid use of PPC MMA type}}
  dmr_t *vq;
};

struct TestDmrPStruct {
  int a;
  float b;
  dmrp_t c; // expected-error {{invalid use of PPC MMA type}}
  dmrp_t *vq;
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

int testDmrPOperators1(int *ptr) {
  __dmr2048 *vdmrpp = (__dmr2048 *)ptr;
  __dmr2048 vdmrp1 = *(vdmrpp + 0);
  __dmr2048 vdmrp2 = *(vdmrpp + 1);
  __dmr2048 vdmrp3 = *(vdmrpp + 2);
  if (vdmrp1) // expected-error {{statement requires expression of scalar type ('__dmr2048' invalid)}}
    *(vdmrpp + 10) = vdmrp1;
  if (!vdmrp2) // expected-error {{invalid argument type '__dmr2048' to unary expression}}
    *(vdmrpp + 11) = vdmrp3;
  int c1 = vdmrp1 && vdmrp2; // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  int c2 = vdmrp2 == vdmrp3; // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  int c3 = vdmrp2 < vdmrp1;  // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  return c1 || c2 || c3;
}

void testDmrPOperators2(int *ptr) {
  __dmr2048 *vdmrpp = (__dmr2048 *)ptr;
  __dmr2048 vdmrp1 = *(vdmrpp + 0);
  __dmr2048 vdmrp2 = *(vdmrpp + 1);
  __dmr2048 vdmrp3 = *(vdmrpp + 2);
  vdmrp1 = -vdmrp1;        // expected-error {{invalid argument type '__dmr2048' to unary expression}}
  vdmrp2 = vdmrp1 + vdmrp3; // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  vdmrp2 = vdmrp2 * vdmrp3; // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  vdmrp3 = vdmrp3 | vdmrp3; // expected-error {{invalid operands to binary expression ('__dmr2048' and '__dmr2048')}}
  vdmrp3 = vdmrp3 << 2;    // expected-error {{invalid operands to binary expression ('__dmr2048' and 'int')}}
  *(vdmrpp + 10) = vdmrp1;
  *(vdmrpp + 11) = vdmrp2;
  *(vdmrpp + 12) = vdmrp3;
}


vector unsigned char testDmrPOperators3(int *ptr) {
  __dmr2048 *vdmrpp = (__dmr2048 *)ptr;
  __dmr2048 vdmrp1 = *(vdmrpp + 0);
  __dmr2048 vdmrp2 = *(vdmrpp + 1);
  __dmr2048 vdmrp3 = *(vdmrpp + 2);
  vdmrp1 ? *(vdmrpp + 10) = vdmrp2 : *(vdmrpp + 11) = vdmrp3; // expected-error {{used type '__dmr2048' where arithmetic or pointer type is required}}
  vdmrp2 = vdmrp3;
  return vdmrp2[1]; // expected-error {{subscripted value is not an array, pointer, or vector}}
}

void testDmrPOperators4(int v, void *ptr) {
  __dmr2048 *vdmrpp = (__dmr2048 *)ptr;
  __dmr2048 vdmrp1 = (__dmr2048)v;   // expected-error {{used type '__dmr2048' where arithmetic or pointer type is required}}
  __dmr2048 vdmrp2 = (__dmr2048)vdmrpp; // expected-error {{used type '__dmr2048' where arithmetic or pointer type is required}}
}
