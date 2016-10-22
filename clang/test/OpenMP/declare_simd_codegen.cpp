// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - -femit-all-decls | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - -femit-all-decls | FileCheck %s
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

#pragma omp declare simd linear(d : 8)
#pragma omp declare simd inbranch simdlen(32)
#pragma omp declare simd notinbranch
void add_1(float *d) {}

#pragma omp declare simd aligned(hp, hp2)
template <class C>
void h(C *hp, C *hp2, C *hq, C *lin) {
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

// CHECK-DAG: "_ZGVbM4l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVbN4l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVcM8l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVcN8l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVdM8l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVdN8l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVeM16l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVeN16l8__Z5add_1Pf"
// CHECK-DAG: "_ZGVbM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVcM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVdM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVeM32v__Z5add_1Pf"
// CHECK-DAG: "_ZGVbN2v__Z5add_1Pf"
// CHECK-DAG: "_ZGVcN4v__Z5add_1Pf"
// CHECK-DAG: "_ZGVdN4v__Z5add_1Pf"
// CHECK-DAG: "_ZGVeN8v__Z5add_1Pf"

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

// CHECK-DAG: "_ZGVbM4uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVbN4uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVcM8uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVcN8uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVdM8uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVdN8uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVeM16uus1__ZN2VV3addEii"
// CHECK-DAG: "_ZGVeN16uus1__ZN2VV3addEii"

// CHECK-DAG: "_ZGVbM4lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVbN4lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVcM8lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVcN8lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVdM8lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVdN8lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVeM16lla16l4a4__ZN2VV6taddpfEPfRS0_"
// CHECK-DAG: "_ZGVeN16lla16l4a4__ZN2VV6taddpfEPfRS0_"

// CHECK-DAG: "_ZGVbM4vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbN4vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcM8vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcN8vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdM8vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdN8vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeM16vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeN16vvl8__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbM4vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVbN4vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcM8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVcN8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdM8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVdN8vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeM16vva8v__ZN2VV4taddERA_iRi"
// CHECK-DAG: "_ZGVeN16vva8v__ZN2VV4taddERA_iRi"

// CHECK-DAG: "_ZGVbM4vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVbN4vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVcM8vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVcN8vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVdM8vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVdN8vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVeM16vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"
// CHECK-DAG: "_ZGVeN16vva32l16a16__ZN3TVVILi16EfE6taddpfEPfRS1_"

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

// CHECK-DAG: "_ZGVbM64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVbN64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVcM64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVcN64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVdM64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVdN64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVeM64va128l64__Z3fooILi64EEvRAT__iRPf"
// CHECK-DAG: "_ZGVeN64va128l64__Z3fooILi64EEvRAT__iRPf"

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

// CHECK-DAG: "_ZGVbM4us2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcM8us2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdM8us2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeM16us2u__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVbM4vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVbN4vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcM8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVcN8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdM8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVdN8vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeM16vvv__Z3bax2VVPdi"
// CHECK-DAG: "_ZGVeN16vvv__Z3bax2VVPdi"

// CHECK-DAG: "_ZGVbM4ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVbN4ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVcM8ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVcN8ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVdM8ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVdN8ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVeM16ua16vl1__Z3fooPffi"
// CHECK-DAG: "_ZGVeN16ua16vl1__Z3fooPffi"

// CHECK-DAG: "_ZGVbN2v__Z3food"
// CHECK-DAG: "_ZGVcN4v__Z3food"
// CHECK-DAG: "_ZGVdN4v__Z3food"
// CHECK-DAG: "_ZGVeN8v__Z3food"

// CHECK-NOT: "_ZGV{{.+}}__Z1fRA_i

#endif
