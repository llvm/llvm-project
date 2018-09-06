// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fexceptions -fcxx-exceptions
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -ferror-limit 100

#ifndef HEADER
#define HEADER

namespace std {
  class type_info;
}

template <typename T>
class TemplateClass {
  T a;
public:
  TemplateClass() { (void)typeid(int); } // expected-error {{use of typeid requires -frtti}}
  T f_method() const { return a; }
};

int foo();

int baz1();

int baz2();

int baz4() { return 5; }

template <typename T>
T FA() {
  TemplateClass<T> s;
  return s.f_method();
}

#pragma omp declare target
struct S {
  int a;
  S(int a) : a(a) { (void)typeid(int); } // expected-error {{use of typeid requires -frtti}}
};

int foo() { return 0; }
int b = 15;
int d;
#pragma omp end declare target
int c;

int bar() { return 1 + foo() + bar() + baz1() + baz2(); }

int maini1() {
  int a;
  static long aa = 32;
#pragma omp target map(tofrom \
                       : a, b)
  {
    S s(a);
    static long aaa = 23;
    a = foo() + bar() + b + c + d + aa + aaa + FA<int>();
    (void)typeid(int); // expected-error {{use of typeid requires -frtti}}
  }
  return baz4();
}

int baz3() { return 2 + baz2(); }
int baz2() {
#pragma omp target
  (void)typeid(int); // expected-error {{use of typeid requires -frtti}}
  return 2 + baz3();
}

#endif // HEADER
