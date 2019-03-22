// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-unknown-linux -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-host.bc -fsyntax-only

struct T {
  char a;
  __float128 f;
  char c;
  T() : a(12), f(15) {}
  T &operator+(T &b) { f += b.a; return *this;} // expected-error {{'__float128' is not supported on this target}}
};

struct T1 {
  char a;
  __int128 f;
  __int128 f1;
  char c;
  T1() : a(12), f(15) {}
  T1 &operator/(T1 &b) { f /= b.a; return *this;}
};

#pragma omp declare target
T a = T();
T f = a;
void foo(T a = T()) {
  a = a + f; // expected-note {{called by 'foo'}}
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
