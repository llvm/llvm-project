// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown \
// RUN:   -verify=host -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc \
// RUN:   %s -o %t-ppc-host.bc -fexceptions -fcxx-exceptions
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple nvptx64-unknown-unknown \
// RUN:   -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s \
// RUN:   -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - \
// RUN:   -fexceptions -fcxx-exceptions -ferror-limit 100

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

int bar() { return 1 + foo() + bar() + baz1() + baz2(); } // expected-note {{called by 'bar'}}

int maini1() {
  int a;
  static long aa = 32;
  try {
#pragma omp target map(tofrom \
                       : a, b)
  {
    // expected-note@+1 {{called by 'maini1'}}
    S s(a);
    static long aaa = 23;
    a = foo() + bar() + b + c + d + aa + aaa + FA<int>(); // expected-note{{called by 'maini1'}}
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

int baz1() { throw 1; } // expected-error {{cannot use 'throw' with exceptions disabled}}

int foobar1();
int foobar2();

int (*A)() = &foobar1;
#pragma omp declare target
int (*B)() = &foobar2;
#pragma omp end declare target

int foobar1() { throw 1; }
int foobar2() { throw 1; } // expected-error {{cannot use 'throw' with exceptions disabled}}


int foobar3();
int (*C)() = &foobar3; // expected-warning {{declaration is not declared in any declare target region}}
                       // host-warning@-1 {{declaration is not declared in any declare target region}}
#pragma omp declare target
int (*D)() = C; // expected-note {{used here}}
                // host-note@-1 {{used here}}
#pragma omp end declare target
int foobar3() { throw 1; }

// Check no infinite recursion in deferred diagnostic emitter.
long E = (long)&E;

#endif // HEADER
