// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc -fexceptions -fcxx-exceptions
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - -fexceptions -fcxx-exceptions -ferror-limit 100

#ifndef HEADER
#define HEADER

template <typename T>
class TemplateClass {
  T a;
public:
  TemplateClass() { throw 1;}
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
  S(int a) : a(a) { throw 1; } // expected-error {{cannot use 'throw' with exceptions disabled}}
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
  try {
#pragma omp target map(tofrom \
                       : a, b)
  {
    S s(a);
    static long aaa = 23;
    a = foo() + bar() + b + c + d + aa + aaa + FA<int>();
    if (!a)
      throw "Error"; // expected-error {{cannot use 'throw' with exceptions disabled}}
  }
  } catch(...) {
  }
  return baz4();
}

int baz3() { return 2 + baz2(); }
int baz2() {
#pragma omp target
  try { // expected-error {{cannot use 'try' with exceptions disabled}}
  ++c;
  } catch (...) {
  }
  return 2 + baz3();
}

#endif // HEADER
