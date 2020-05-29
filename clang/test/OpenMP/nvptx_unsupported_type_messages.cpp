// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -aux-triple powerpc64le-unknown-linux-gnu -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only

struct T {
  char a;
#ifndef _ARCH_PPC
  // expected-note@+1 {{'f' defined here}}
  __float128 f;
#else
  // expected-note@+1 {{'f' defined here}}
  long double f;
#endif
  char c;
  T() : a(12), f(15) {}
#ifndef _ARCH_PPC
// expected-error@+5 {{'f' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#else
// expected-error@+3 {{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#endif
  T &operator+(T &b) {
    f += b.a;
    return *this;
  }
};

struct T1 {
  char a;
  __int128 f;
  __int128 f1;
  char c;
  T1() : a(12), f(15) {}
  T1 &operator/(T1 &b) {
    f /= b.a;
    return *this;
  }
};

#ifndef _ARCH_PPC
// expected-note@+1 {{'boo' defined here}}
void boo(__float128 A) { return; }
#else
// expected-note@+1 {{'boo' defined here}}
void boo(long double A) { return; }
#endif
#pragma omp declare target
T a = T();
T f = a;
void foo(T a = T()) {
  a = a + f; // expected-note {{called by 'foo'}}
#ifndef _ARCH_PPC
// expected-error@+4 {{'boo' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#else
// expected-error@+2 {{'boo' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
#endif
  boo(0);
  return;
}
T bar() {
  return T();
}

void baz() {
  T t = bar();
}
T1 a1 = T1();
T1 f1 = a1;
void foo1(T1 a = T1()) {
  a = a / f1;
  return;
}
T1 bar1() {
  return T1();
}
void baz1() {
  T1 t = bar1();
}
#pragma omp end declare target

#ifndef _ARCH_PPC
// expected-note@+1 3{{'f' defined here}}
__float128 foo1(__float128 f) {
#pragma omp target map(f)
  // expected-error@+1 3{{'f' requires 128 bit size '__float128' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}
#else
// expected-note@+1 3{{'f' defined here}}
long double foo1(long double f) {
#pragma omp target map(f)
  // expected-error@+1 3{{'f' requires 128 bit size 'long double' type support, but device 'nvptx64-unknown-unknown' does not support it}}
  f = 1;
  return f;
}
#endif

T foo3() {
  T S;
#pragma omp target map(S)
  S.a = 1;
  return S;
}

// Allow all sorts of stuff on host
#ifndef _ARCH_PPC
__float128 q, b;
__float128 c = q + b;
#else
long double q, b;
long double c = q + b;
#endif

void hostFoo() {
  boo(c - b);
}

long double qa, qb;
decltype(qa + qb) qc;
double qd[sizeof(-(-(qc * 2)))];
