// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu pwr10 %s -verify
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu pwr10 %s -verify
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

// vector quad

// typedef
typedef __vector_quad vq_t;

// function argument
void testVQArg1(__vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

void testVQArg2(const __vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

void testVQArg6(const vq_t vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

// function return
__vector_quad testVQRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  return *vqp; // expected-error {{invalid use of PPC MMA type}}
}

const vq_t testVQRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  return *vqp; // expected-error {{invalid use of PPC MMA type}}
}

// global
__vector_quad globalvq;        // expected-error {{invalid use of PPC MMA type}}
const __vector_quad globalvq2; // expected-error {{invalid use of PPC MMA type}}
__vector_quad *globalvqp;
const __vector_quad *const globalvqp2;
vq_t globalvq_t; // expected-error {{invalid use of PPC MMA type}}


// struct field
struct TestVQStruct {
  int a;
  float b;
  __vector_quad c; // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vq;
};

// operators
int testVQOperators1(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq1 = *(vqp + 0);
  __vector_quad vq2 = *(vqp + 1);
  __vector_quad vq3 = *(vqp + 2);
  if (vq1) // expected-error {{statement requires expression of scalar type ('__vector_quad' invalid)}}
    *(vqp + 10) = vq1;
  if (!vq2) // expected-error {{invalid argument type '__vector_quad' to unary expression}}
    *(vqp + 11) = vq3;
  int c1 = vq1 && vq2; // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  int c2 = vq2 == vq3; // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  int c3 = vq2 < vq1;  // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  return c1 || c2 || c3;
}

void testVQOperators2(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq1 = *(vqp + 0);
  __vector_quad vq2 = *(vqp + 1);
  __vector_quad vq3 = *(vqp + 2);
  vq1 = -vq1;      // expected-error {{invalid argument type '__vector_quad' to unary expression}}
  vq2 = vq1 + vq3; // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  vq2 = vq2 * vq3; // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  vq3 = vq3 | vq3; // expected-error {{invalid operands to binary expression ('__vector_quad' and '__vector_quad')}}
  vq3 = vq3 << 2;  // expected-error {{invalid operands to binary expression ('__vector_quad' and 'int')}}
  *(vqp + 10) = vq1;
  *(vqp + 11) = vq2;
  *(vqp + 12) = vq3;
}

vector unsigned char testVQOperators3(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq1 = *(vqp + 0);
  __vector_quad vq2 = *(vqp + 1);
  __vector_quad vq3 = *(vqp + 2);
  vq1 ? *(vqp + 10) = vq2 : *(vqp + 11) = vq3; // expected-error {{used type '__vector_quad' where arithmetic or pointer type is required}}
  vq2 = vq3;
  return vq2[1]; // expected-error {{subscripted value is not an array, pointer, or vector}}
}

void testVQOperators4(int v, void *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq1 = (__vector_quad)v;   // expected-error {{used type '__vector_quad' where arithmetic or pointer type is required}}
  __vector_quad vq2 = (__vector_quad)vqp; // expected-error {{used type '__vector_quad' where arithmetic or pointer type is required}}
}

// vector pair

// typedef
typedef __vector_pair vp_t;

// function argument
void testVPArg1(__vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

void testVPArg2(const __vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

void testVPArg6(const vp_t vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

// function return
__vector_pair testVPRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  return *vpp; // expected-error {{invalid use of PPC MMA type}}
}

const vp_t testVPRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  return *vpp; // expected-error {{invalid use of PPC MMA type}}
}

// global
__vector_pair globalvp;        // expected-error {{invalid use of PPC MMA type}}
const __vector_pair globalvp2; // expected-error {{invalid use of PPC MMA type}}
__vector_pair *globalvpp;
const __vector_pair *const globalvpp2;
vp_t globalvp_t; // expected-error {{invalid use of PPC MMA type}}

// struct field
struct TestVPStruct {
  int a;
  float b;
  __vector_pair c; // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vp;
};

// operators
int testVPOperators1(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp1 = *(vpp + 0);
  __vector_pair vp2 = *(vpp + 1);
  __vector_pair vp3 = *(vpp + 2);
  if (vp1) // expected-error {{statement requires expression of scalar type ('__vector_pair' invalid)}}
    *(vpp + 10) = vp1;
  if (!vp2) // expected-error {{invalid argument type '__vector_pair' to unary expression}}
    *(vpp + 11) = vp3;
  int c1 = vp1 && vp2; // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  int c2 = vp2 == vp3; // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  int c3 = vp2 < vp1;  // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  return c1 || c2 || c3;
}

void testVPOperators2(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp1 = *(vpp + 0);
  __vector_pair vp2 = *(vpp + 1);
  __vector_pair vp3 = *(vpp + 2);
  vp1 = -vp1;      // expected-error {{invalid argument type '__vector_pair' to unary expression}}
  vp2 = vp1 + vp3; // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  vp2 = vp2 * vp3; // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  vp3 = vp3 | vp3; // expected-error {{invalid operands to binary expression ('__vector_pair' and '__vector_pair')}}
  vp3 = vp3 << 2;  // expected-error {{invalid operands to binary expression ('__vector_pair' and 'int')}}
  *(vpp + 10) = vp1;
  *(vpp + 11) = vp2;
  *(vpp + 12) = vp3;
}

vector unsigned char testVPOperators3(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp1 = *(vpp + 0);
  __vector_pair vp2 = *(vpp + 1);
  __vector_pair vp3 = *(vpp + 2);
  vp1 ? *(vpp + 10) = vp2 : *(vpp + 11) = vp3; // expected-error {{used type '__vector_pair' where arithmetic or pointer type is required}}
  vp2 = vp3;
  return vp2[1]; // expected-error {{subscripted value is not an array, pointer, or vector}}
}

void testVPOperators4(int v, void *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp1 = (__vector_pair)v;   // expected-error {{used type '__vector_pair' where arithmetic or pointer type is required}}
  __vector_pair vp2 = (__vector_pair)vpp; // expected-error {{used type '__vector_pair' where arithmetic or pointer type is required}}
}

void testBuiltinTypes1(const __vector_pair *vpp, const __vector_pair *vp2, float f) {
  __vector_pair vp = __builtin_vsx_lxvp(f, vpp); // expected-error {{passing 'float' to parameter of incompatible type 'long'}}
  __builtin_vsx_stxvp(vp, 32799, vp2);           // expected-error {{passing 'int' to parameter of incompatible type 'long'}}
}

void testBuiltinTypes2(__vector_pair *vpp, const __vector_pair *vp2, unsigned char c) {
  __vector_pair vp = __builtin_vsx_lxvp(6L, vpp); // expected-error {{passing '__vector_pair *' to parameter of incompatible type 'const __vector_pair *'}}
  __builtin_vsx_stxvp(vp, c, vp2);                // expected-error {{passing 'unsigned char' to parameter of incompatible type 'long'}}
}

void testBuiltinTypes3(vector int v, __vector_pair *vp2, signed long l, unsigned short s) {
  __vector_pair vp = __builtin_vsx_lxvp(l, v); // expected-error {{passing '__vector int' (vector of 4 'int' values) to parameter of incompatible type 'const __vector_pair *'}}
  __builtin_vsx_stxvp(vp, l, s);               // expected-error {{passing 'unsigned short' to parameter of incompatible type '__vector_pair *'}}
}

void testRestrictQualifiedPointer1(int *__restrict acc) {
  vector float arr[4];
  __builtin_mma_disassemble_acc(arr, acc); // expected-error {{passing 'int *restrict' to parameter of incompatible type '__vector_quad *'}}
}

void testVolatileQualifiedPointer1(int *__volatile acc) {
  vector float arr[4];
  __builtin_mma_disassemble_acc(arr, acc); // expected-error {{passing 'int *volatile' to parameter of incompatible type '__vector_quad *'}}
}
