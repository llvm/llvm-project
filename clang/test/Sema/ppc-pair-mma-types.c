// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -fsyntax-only \
// RUN:   -target-cpu future %s -verify

// The use of PPC MMA types is strongly restricted. Non-pointer MMA variables
// can only be declared in functions and a limited number of operations are
// supported on these types. This test case checks that invalid uses of MMA
// types are correctly prevented.

// vector quad

// typedef
typedef __vector_quad vq_t;
void testVQTypedef(int *inp, int *outp) {
  vq_t *vqin = (vq_t *)inp;
  vq_t *vqout = (vq_t *)outp;
  *vqout = *vqin;
}

// function argument
void testVQArg1(__vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

void testVQArg2(const __vector_quad vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

void testVQArg3(__vector_quad *vq, int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = *vq;
}

void testVQArg4(const __vector_quad *const vq, int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = *vq;
}

void testVQArg5(__vector_quad vqa[], int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vqa[0];
}

void testVQArg6(const vq_t vq, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = vq;
}

void testVQArg7(const vq_t *vq, int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  *vqp = *vq;
}

// function return
__vector_quad testVQRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  return *vqp; // expected-error {{invalid use of PPC MMA type}}
}

__vector_quad *testVQRet2(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  return vqp + 2;
}

const __vector_quad *testVQRet3(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  return vqp + 2;
}

const vq_t testVQRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vqp = (__vector_quad *)ptr;
  return *vqp; // expected-error {{invalid use of PPC MMA type}}
}

const vq_t *testVQRet5(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  return vqp + 2;
}

// global
__vector_quad globalvq;        // expected-error {{invalid use of PPC MMA type}}
const __vector_quad globalvq2; // expected-error {{invalid use of PPC MMA type}}
__vector_quad *globalvqp;
const __vector_quad *const globalvqp2;
vq_t globalvq_t; // expected-error {{invalid use of PPC MMA type}}

// local
void testVQLocal(int *ptr, vector unsigned char vc) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq1 = *vqp;
  __vector_quad vq2;
  __builtin_mma_xxsetaccz(&vq2);
  __vector_quad vq3;
  __builtin_mma_xvi4ger8(&vq3, vc, vc);
  *vqp = vq3;
}

// struct field
struct TestVQStruct {
  int a;
  float b;
  __vector_quad c; // expected-error {{invalid use of PPC MMA type}}
  __vector_quad *vq;
};

// sizeof / alignof
int testVQSizeofAlignof(int *ptr) {
  __vector_quad *vqp = (__vector_quad *)ptr;
  __vector_quad vq = *vqp;
  unsigned sizet = sizeof(__vector_quad);
  unsigned alignt = __alignof__(__vector_quad);
  unsigned sizev = sizeof(vq);
  unsigned alignv = __alignof__(vq);
  return sizet + alignt + sizev + alignv;
}

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
void testVPTypedef(int *inp, int *outp) {
  vp_t *vpin = (vp_t *)inp;
  vp_t *vpout = (vp_t *)outp;
  *vpout = *vpin;
}

// function argument
void testVPArg1(__vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

void testVPArg2(const __vector_pair vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

void testVPArg3(__vector_pair *vp, int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = *vp;
}

void testVPArg4(const __vector_pair *const vp, int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = *vp;
}

void testVPArg5(__vector_pair vpa[], int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vpa[0];
}

void testVPArg6(const vp_t vp, int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = vp;
}

void testVPArg7(const vp_t *vp, int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  *vpp = *vp;
}

// function return
__vector_pair testVPRet1(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  return *vpp; // expected-error {{invalid use of PPC MMA type}}
}

__vector_pair *testVPRet2(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  return vpp + 2;
}

const __vector_pair *testVPRet3(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  return vpp + 2;
}

const vp_t testVPRet4(int *ptr) { // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vpp = (__vector_pair *)ptr;
  return *vpp; // expected-error {{invalid use of PPC MMA type}}
}

const vp_t *testVPRet5(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  return vpp + 2;
}

// global
__vector_pair globalvp;        // expected-error {{invalid use of PPC MMA type}}
const __vector_pair globalvp2; // expected-error {{invalid use of PPC MMA type}}
__vector_pair *globalvpp;
const __vector_pair *const globalvpp2;
vp_t globalvp_t; // expected-error {{invalid use of PPC MMA type}}

// local
void testVPLocal(int *ptr, vector unsigned char vc) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp1 = *vpp;
  __vector_pair vp2;
  __builtin_vsx_assemble_pair(&vp2, vc, vc);
  __vector_pair vp3;
  __vector_quad vq;
  __builtin_mma_xvf64ger(&vq, vp3, vc);
  *vpp = vp3;
}

// struct field
struct TestVPStruct {
  int a;
  float b;
  __vector_pair c; // expected-error {{invalid use of PPC MMA type}}
  __vector_pair *vp;
};

// sizeof / alignof
int testVPSizeofAlignof(int *ptr) {
  __vector_pair *vpp = (__vector_pair *)ptr;
  __vector_pair vp = *vpp;
  unsigned sizet = sizeof(__vector_pair);
  unsigned alignt = __alignof__(__vector_pair);
  unsigned sizev = sizeof(vp);
  unsigned alignv = __alignof__(vp);
  return sizet + alignt + sizev + alignv;
}

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
  __vector_pair vp = __builtin_vsx_lxvp(f, vpp); // expected-error {{passing 'float' to parameter of incompatible type 'long long'}}
  __builtin_vsx_stxvp(vp, 32799, vp2);           // expected-error {{passing 'int' to parameter of incompatible type 'long long'}}
}

void testBuiltinTypes2(__vector_pair *vpp, const __vector_pair *vp2, unsigned char c) {
  __vector_pair vp = __builtin_vsx_lxvp(6LL, vpp); // expected-error {{passing '__vector_pair *' to parameter of incompatible type 'const __vector_pair *'}}
  __builtin_vsx_stxvp(vp, c, vp2);                 // expected-error {{passing 'unsigned char' to parameter of incompatible type 'long long'}}
}

void testBuiltinTypes3(vector int v, __vector_pair *vp2, signed long long ll, unsigned short s) {
  __vector_pair vp = __builtin_vsx_lxvp(ll, v); // expected-error {{passing '__vector int' (vector of 4 'int' values) to parameter of incompatible type 'const __vector_pair *'}}
  __builtin_vsx_stxvp(vp, ll, s);               // expected-error {{passing 'unsigned short' to parameter of incompatible type 'const __vector_pair *'}}
}
