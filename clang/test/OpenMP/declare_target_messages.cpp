// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -fnoopenmp-use-tls -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -fnoopenmp-use-tls -ferror-limit 100 -o - %s

#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

int a, b; // expected-warning {{declaration is not declared in any declare target region}}
__thread int t; // expected-note {{defined as threadprivate or thread local}}

#pragma omp declare target . // expected-error {{expected '(' after 'declare target'}}

#pragma omp declare target
void f();
#pragma omp end declare target shared(a) // expected-warning {{extra tokens at the end of '#pragma omp end declare target' are ignored}}

#pragma omp declare target map(a) // expected-error {{unexpected 'map' clause, only 'to' or 'link' clauses expected}}

#pragma omp declare target to(foo1) // expected-error {{use of undeclared identifier 'foo1'}}

#pragma omp declare target link(foo2) // expected-error {{use of undeclared identifier 'foo2'}}

void c(); // expected-warning {{declaration is not declared in any declare target region}}

void func() {} // expected-note {{'func' defined here}}

#pragma omp declare target link(func) // expected-error {{function name is not allowed in 'link' clause}}

extern int b;

struct NonT {
  int a;
};

typedef int sint;

#pragma omp declare target // expected-note {{to match this '#pragma omp declare target'}}
#pragma omp threadprivate(a) // expected-note {{defined as threadprivate or thread local}}
extern int b;
int g;

struct T {
  int a;
  virtual int method();
};

class VC {
  T member;
  NonT member1;
  public:
    virtual int method() { T a; return 0; }
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
  VC object1;
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
#pragma omp declare target
  int v;
#pragma omp end declare target
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

#pragma omp declare target link(S) // expected-error {{'S' used in declare target directive is not a variable or a function name}}

#pragma omp declare target (x, x) // expected-error {{'x' appears multiple times in clauses on the same declare target directive}}
#pragma omp declare target to(x) to(x) // expected-error {{'x' appears multiple times in clauses on the same declare target directive}}
#pragma omp declare target link(x) // expected-error {{'x' must not appear in both clauses 'to' and 'link'}}

#pragma omp declare target // expected-error {{expected '#pragma omp end declare target'}} expected-note {{to match this '#pragma omp declare target'}}
