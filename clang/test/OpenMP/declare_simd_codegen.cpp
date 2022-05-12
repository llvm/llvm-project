// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - -femit-all-decls | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls | FileCheck %s

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - -femit-all-decls | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

void add_1(float *d);

#pragma omp declare simd linear(d : 8)
#pragma omp declare simd inbranch simdlen(32)
#pragma omp declare simd notinbranch
void add_1(float *d);

#pragma omp declare simd linear(d : 8)
#pragma omp declare simd inbranch simdlen(32)
#pragma omp declare simd notinbranch
void add_1(float *d) {}

void add_1(float *d);

#pragma omp declare simd linear(d : 8)
#pragma omp declare simd inbranch simdlen(32)
#pragma omp declare simd notinbranch
void add_2(float *d);

#pragma omp declare simd aligned(hp, hp2)
template <class C>
void h(C *hp, C *hp2, C *hq, C *lin) {
  add_2(0);
}

// Explicit specialization with <C=int>.
// Pragmas need to be same, otherwise standard says that's undefined behavior.
#pragma omp declare simd aligned(hp, hp2)
template <>
void h(int *hp, int *hp2, int *hq, int *lin) {
  // Implicit specialization with <C=float>.
  // This is special case where the directive is stored by Sema and is
  // generated together with the (pending) function instatiation.
  h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
}

class VV {
public:
#pragma omp declare simd uniform(this, a) linear(val(b) : a)
  int add(int a, int b) __attribute__((cold)) { return a + b; }

#pragma omp declare simd aligned(b : 4) aligned(a) linear(ref(b) : 4) linear(this, a)
  float taddpf(float *a, float *&b) { return *a + *b; }

#pragma omp declare simd linear(uval(c) : 8)
#pragma omp declare simd aligned(b : 8)
  int tadd(int (&b)[], int &c) { return x[b[0]] + b[0]; }

private:
  int x[10];
} vv;

template <int X, typename T>
class TVV {
public:
#pragma omp declare simd simdlen(X)
  int tadd(int a, int b) { return a + b; }

#pragma omp declare simd aligned(a : X * 2) aligned(b) linear(ref(b) : X)
  float taddpf(float *a, T *&b) { return *a + *b; }

#pragma omp declare simd
#pragma omp declare simd uniform(this, b)
  int tadd(int b) { return x[b] + b; }

private:
  int x[X];
};

#pragma omp declare simd simdlen(N) aligned(b : N * 2) linear(uval(c) : N)
template <int N>
void foo(int (&b)[N], float *&c) {}

TVV<16, float> t16;

void f(int (&g)[]) {
  float a = 1.0f, b = 2.0f;
  float *p = &b;
  float r = t16.taddpf(&a, p);
  int res = t16.tadd(b);
  int c[64];
  vv.add(res, res);
  vv.taddpf(p, p);
  vv.tadd(g, res);
  foo(c, p);
}

#pragma omp declare simd
#pragma omp declare simd notinbranch aligned(a : 32)
int bar(VV v, float *a) { return 0; }
#pragma omp declare simd
#pragma omp declare simd notinbranch aligned(a)
float baz(VV v, int a[]) { return 0; }
#pragma omp declare simd
#pragma omp declare simd notinbranch aligned(a)
double bay(VV v, double *&a) { return 0; }
#pragma omp declare simd
#pragma omp declare simd inbranch linear(a : b) uniform(v, b)
void bax(VV v, double *a, int b) {}
#pragma omp declare simd uniform(q) aligned(q : 16) linear(k : 1)
float foo(float *q, float x, int k) { return 0; }
#pragma omp declare simd notinbranch
double foo(double x) { return 0; }

#pragma omp declare simd notinbranch linear(i)
double constlinear(const int i) { return 0.0; }

// Test linear modifiers
// linear(x) cases
#pragma omp declare simd simdlen(4) linear(a:2) linear(b:4) linear(c:8) \
                                    linear(d,e,f)
double One(int &a, int *b, int c, int &d, int *e, int f) {
  return a + *b + c;
}

// linear(val(x)) cases
#pragma omp declare simd simdlen(4) linear(val(a):2) linear(val(b):4) \
                                    linear(val(c):8) linear(val(d,e,f))
double Two(int &a, int *b, int c, int &d, int *e, int f) {
  return a + *b + c;
}

// linear(uval(x) case
#pragma omp declare simd simdlen(4) linear(uval(a):2) linear(uval(b))
double Three(int &a, int &b) {
  return a;
}

// linear(ref(x) case
#pragma omp declare simd simdlen(4) linear(ref(a):2) linear(ref(b))
double Four(int& a, int &b) {
  return a;
}

// Test reference parameters with variable stride.
#pragma omp declare simd simdlen(4) uniform(a)               \
                         linear(b:2) linear(c:a)             \
                         linear(val(d):4) linear(val(e):a)   \
                         linear(uval(f):8) linear(uval(g):a) \
                         linear(ref(h):16) linear(ref(i):a)
double Five(int a, short &b, short &c, short &d, short &e, short &f, short &g,
            short &h, short &i) {
  return a + int(b);
}

// Test negative strides
#pragma omp declare simd simdlen(4) linear(a:-2) linear(b:-8) \
                                    linear(uval(c):-4) linear(ref(d):-16) \
                                    linear(e:-1) linear(f:-1) linear(g:0)
double Six(int a, float *b, int &c, int *&d, char e, char *f, short g) {
 return a + int(*b) + c + *d + e + *f + g;
}

// CHECK-DAG: define {{.+}}@_Z5add_1Pf(
// CHECK-DAG: define {{.+}}@_Z1hIiEvPT_S1_S1_S1_(
// CHECK-DAG: define {{.+}}@_Z1hIfEvPT_S1_S1_S1_(
// CHECK-DAG: define {{.+}}@_ZN2VV3addEii(
// CHECK-DAG: define {{.+}}@_ZN2VV6taddpfEPfRS0_(
// CHECK-DAG: define {{.+}}@_ZN2VV4taddERA_iRi(
// CHECK-DAG: define {{.+}}@_Z1fRA_i(
// CHECK-DAG: define {{.+}}@_ZN3TVVILi16EfE6taddpfEPfRS1_(
// CHECK-DAG: define {{.+}}@_ZN3TVVILi16EfE4taddEi(
// CHECK-DAG: define {{.+}}@_Z3fooILi64EEvRAT__iRPf(
// CHECK-DAG: define {{.+}}@_Z3bar2VVPf(
// CHECK-DAG: define {{.+}}@_Z3baz2VVPi(
// CHECK-DAG: define {{.+}}@_Z3bay2VVRPd(
// CHECK-DAG: define {{.+}}@_Z3bax2VVPdi(
// CHECK-DAG: define {{.+}}@_Z3fooPffi(
// CHECK-DAG: define {{.+}}@_Z3food(
// CHECK-DAG: declare {{.+}}@_Z5add_2Pf(
// CHECK-DAG: define {{.+}}@_Z11constlineari(
// CHECK-DAG: define {{.+}}@_Z3OneRiPiiS_S0_i
// CHECK-DAG: define {{.+}}@_Z3TwoRiPiiS_S0_i
// CHECK-DAG: define {{.+}}@_Z5ThreeRiS_
// CHECK-DAG: define {{.+}}@_Z4FourRiS_
// CHECK-DAG: define {{.+}}@_Z4FiveiRsS_S_S_S_S_S_S_
// CHECK-DAG: define {{.+}}@_Z3SixiPfRiRPicPcs

// CHECK-DAG: "_ZGVbM4l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVbN4l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVcM8l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVcN8l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVdM8l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVdN8l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVeM16l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVeN16l32__Z5add_1Pf"
// CHECK-DAG: "_ZGVbM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVcM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVdM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVeM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVbN2v__Z5add_1Pf"
// CHECK-DAG: "_ZGVcN4v__Z5add_1Pf"
// CHECK-DAG: "_ZGVdN4v__Z5add_1Pf"
// CHECK-DAG: "_ZGVeN8v__Z5add_1Pf"

// CHECK-NOT: _ZGVbN2vv__Z5add_1Pf
// CHECK-NOT: _ZGVcN4vv__Z5add_1Pf
// CHECK-NOT: _ZGVdN4vv__Z5add_1Pf
// CHECK-NOT: _ZGVeN8vv__Z5add_1Pf
// CHECK-NOT: _ZGVbM32vv__Z5add_1Pf
// CHECK-NOT: _ZGVcM32vv__Z5add_1Pf
// CHECK-NOT: _ZGVdM32vv__Z5add_1Pf
// CHECK-NOT: _ZGVeM32vv__Z5add_1Pf
// CHECK-NOT: _ZGVbN4l32v__Z5add_1Pf
// CHECK-NOT: _ZGVcN8l32v__Z5add_1Pf
// CHECK-NOT: _ZGVdN8l32v__Z5add_1Pf
// CHECK-NOT: _ZGVeN16l32v__Z5add_1Pf
// CHECK-NOT: _ZGVbM4l32v__Z5add_1Pf
// CHECK-NOT: _ZGVcM8l32v__Z5add_1Pf
// CHECK-NOT: _ZGVdM8l32v__Z5add_1Pf
// CHECK-NOT: _ZGVeM16l32v__Z5add_1Pf

// CHECK-DAG: "_ZGVbM2va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVbN2va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVcM4va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVcN4va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVdM4va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVdN4va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVeM8va16va16vv__Z1hIiEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVeN8va16va16vv__Z1hIiEvPT_S1_S1_S1_"

// CHECK-DAG: "_ZGVbM2va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVbN2va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVcM4va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVcN4va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVdM4va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVdN4va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVeM8va16va16vv__Z1hIfEvPT_S1_S1_S1_"
// CHECK-DAG: "_ZGVeN8va16va16vv__Z1hIfEvPT_S1_S1_S1_"

// CHECK-DAG: "_ZGVbM4uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVbN4uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVcM8uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVcN8uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVdM8uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVdN8uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVeM16uuls1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVeN16uuls1__ZN2VV3addEii"

// CHECK-DAG: "_ZGVbM4l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVbN4l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVcM8l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVcN8l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVdM8l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVdN8l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVeM16l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVeN16l40l4a16R32a4__ZN2VV6taddpfEPfRS0_"

// CHECK-DAG: "_ZGVbM4vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbN4vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcM8vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcN8vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdM8vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdN8vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeM16vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeN16vvU8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbM4vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbN4vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcM8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcN8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdM8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdN8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeM16vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeN16vva8v__ZN2VV4taddERA_iRi"

// CHECK-DAG: "_ZGVbM4vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVbN4vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVcM8vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVcN8vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVdM8vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVdN8vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVeM16vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVeN16vva32R128a16__ZN3TVVILi16EfE6taddpfEPfRS1_"

// CHECK-DAG: "_ZGVbM4uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVbN4uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVcM8uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVcN8uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVdM8uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVdN8uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVeM16uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVeN16uu__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVbM4vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVbN4vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVcM8vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVcN8vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVdM8vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVdN8vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVeM16vv__ZN3TVVILi16EfE4taddEi"
// CHECK-DAG: "_ZGVeN16vv__ZN3TVVILi16EfE4taddEi"

// CHECK-DAG: "_ZGVbM64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVbN64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVcM64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVcN64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVdM64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVdN64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVeM64va128U64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVeN64va128U64__Z3fooILi64EEvRAT__iRPf"

// CHECK-DAG: "_ZGVbM4vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVbN4vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVcM8vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVcN8vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVdM8vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVdN8vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVeM16vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVeN16vv__Z3bar2VVPf"
// CHECK-DAG: "_ZGVbN4vva32__Z3bar2VVPf"
// CHECK-DAG: "_ZGVcN8vva32__Z3bar2VVPf"
// CHECK-DAG: "_ZGVdN8vva32__Z3bar2VVPf"
// CHECK-DAG: "_ZGVeN16vva32__Z3bar2VVPf"

// CHECK-DAG: "_ZGVbM4vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVbN4vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVcM8vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVcN8vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVdM8vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVdN8vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVeM16vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVeN16vv__Z3baz2VVPi"
// CHECK-DAG: "_ZGVbN4vva16__Z3baz2VVPi"
// CHECK-DAG: "_ZGVcN8vva16__Z3baz2VVPi"
// CHECK-DAG: "_ZGVdN8vva16__Z3baz2VVPi"
// CHECK-DAG: "_ZGVeN16vva16__Z3baz2VVPi"

// CHECK-DAG: "_ZGVbM2vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVbN2vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVcM4vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVcN4vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVdM4vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVdN4vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVeM8vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVeN8vv__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVbN2vva16__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVcN4vva16__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVdN4vva16__Z3bay2VVRPd"
// CHECK-DAG: "_ZGVeN8vva16__Z3bay2VVRPd"

// CHECK-DAG: "_ZGVbM4uls2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcM8uls2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdM8uls2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeM16uls2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVbM4vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVbN4vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcM8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcN8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdM8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdN8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeM16vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeN16vvv__Z3bax2VVPdi"

// CHECK-DAG: "_ZGVbM4ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVbN4ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVcM8ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVcN8ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVdM8ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVdN8ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVeM16ua16vl__Z3fooPffi"
// CHECK-DAG: "_ZGVeN16ua16vl__Z3fooPffi"

// CHECK-DAG: "_ZGVbM4l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVbN4l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVcM8l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVcN8l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVdM8l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVdN8l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVeM16l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVeN16l32__Z5add_2Pf"
// CHECK-DAG: "_ZGVbM32v__Z5add_2Pf"
// CHECK-DAG: "_ZGVcM32v__Z5add_2Pf"
// CHECK-DAG: "_ZGVdM32v__Z5add_2Pf"
// CHECK-DAG: "_ZGVeM32v__Z5add_2Pf"
// CHECK-DAG: "_ZGVbN2v__Z5add_2Pf"
// CHECK-DAG: "_ZGVcN4v__Z5add_2Pf"
// CHECK-DAG: "_ZGVdN4v__Z5add_2Pf"
// CHECK-DAG: "_ZGVeN8v__Z5add_2Pf"

// CHECK-DAG: "_ZGVbN2v__Z3food"
// CHECK-DAG: "_ZGVcN4v__Z3food"
// CHECK-DAG: "_ZGVdN4v__Z3food"
// CHECK-DAG: "_ZGVeN8v__Z3food"

// CHECK-DAG: "_ZGVbN2l__Z11constlineari"
// CHECK-DAG: "_ZGVcN4l__Z11constlineari"
// CHECK-DAG: "_ZGVdN4l__Z11constlineari"
// CHECK-DAG: "_ZGVeN8l__Z11constlineari"

// CHECK-DAG: "_ZGVbM4L2l16l8Ll4l__Z3OneRiPiiS_S0_i"
// CHECK-DAG: "_ZGVbN4L2l16l8Ll4l__Z3OneRiPiiS_S0_i"
// CHECK-DAG: "_ZGVbM4L2l16l8Ll4l__Z3TwoRiPiiS_S0_i"
// CHECK-DAG: "_ZGVbN4L2l16l8Ll4l__Z3TwoRiPiiS_S0_i"
// CHECK-DAG: "_ZGVbM4U2U__Z5ThreeRiS_"
// CHECK-DAG: "_ZGVbN4U2U__Z5ThreeRiS_"
// CHECK-DAG: "_ZGVbM4R8R4__Z4FourRiS_"
// CHECK-DAG: "_ZGVbN4R8R4__Z4FourRiS_"
// CHECK-DAG: "_ZGVbM4uL2Ls0L4Ls0U8Us0R32Rs0__Z4FiveiRsS_S_S_S_S_S_S_"
// CHECK-DAG: "_ZGVbN4uL2Ls0L4Ls0U8Us0R32Rs0__Z4FiveiRsS_S_S_S_S_S_S_"
// CHECK-DAG: "_ZGVbM4ln2ln32Un4Rn128ln1ln1l0__Z3SixiPfRiRPicPcs"
// CHECK-DAG: "_ZGVbN4ln2ln32Un4Rn128ln1ln1l0__Z3SixiPfRiRPicPcs"

// CHECK-NOT: "_ZGV{{.+}}__Z1fRA_i

#endif
