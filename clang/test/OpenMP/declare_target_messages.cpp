// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fnoopenmp-use-tls -ferror-limit 100 -o - %s

#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

int a, b; // expected-warning {{declaration is not declared in any declare target region}}
__thread int t; // expected-note {{defined as threadprivate or thread local}}
#pragma omp declare target private(a) // expected-warning {{extra tokens at the end of '#pragma omp declare target' are ignored}}
void f();
#pragma omp end declare target shared(a) // expected-warning {{extra tokens at the end of '#pragma omp end declare target' are ignored}}
void c(); // expected-warning {{declaration is not declared in any declare target region}}

extern int b;

struct NonT {
  int a;
};

typedef int sint;

#pragma omp declare target // expected-note {{to match this '#pragma omp declare target'}}
#pragma omp threadprivate(a) // expected-note {{defined as threadprivate or thread local}}
extern int b;
int g;

struct T { // expected-note {{mappable type cannot be polymorphic}}
  int a;
  virtual int method();
};

class VC { // expected-note {{mappable type cannot be polymorphic}}
  T member;
  NonT member1;
  public:
    virtual int method() { T a; return 0; } // expected-error {{type 'T' is not mappable to target}}
};

struct C {
  NonT a;
  sint b;
  int method();
  int method1();
};

int C::method1() {
  return 0;
}

void foo() {
  a = 0; // expected-error {{threadprivate variables cannot be used in target constructs}}
  b = 0; // expected-note {{used here}}
  t = 1; // expected-error {{threadprivate variables cannot be used in target constructs}}
  C object;
  VC object1; // expected-error {{type 'VC' is not mappable to target}}
  g = object.method();
  g += object.method1();
  g += object1.method();
  f();
  c(); // expected-note {{used here}}
}
#pragma omp declare target // expected-error {{expected '#pragma omp end declare target'}}
void foo1() {}
#pragma omp end declare target
#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

int C::method() {
  return 0;
}

struct S {
#pragma omp declare target // expected-error {{directive must be at file or namespace scope}}
  int v;
#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}
};

int main (int argc, char **argv) {
#pragma omp declare target // expected-error {{unexpected OpenMP directive '#pragma omp declare target'}}
  int v;
#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}
  foo();
  return (0);
}

namespace {
#pragma omp declare target // expected-note {{to match this '#pragma omp declare target'}}
  int x;
} //  expected-error {{expected '#pragma omp end declare target'}}
#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

#pragma omp declare target // expected-error {{expected '#pragma omp end declare target'}} expected-note {{to match this '#pragma omp declare target'}}
