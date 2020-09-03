// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp45 -fopenmp -fopenmp-version=45 -fnoopenmp-use-tls -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp5,host5 -fopenmp -fopenmp-version=50 -fnoopenmp-use-tls -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp5,dev5 -fopenmp -fopenmp-is-device -fopenmp-targets=x86_64-apple-macos10.7.0 -aux-triple x86_64-apple-macos10.7.0 -fopenmp-version=50 -fnoopenmp-use-tls -ferror-limit 100 -o - %s

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp5,host5 -fopenmp-simd -fopenmp-version=50 -fnoopenmp-use-tls -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp5,host5 -fopenmp-simd -fopenmp-is-device -fopenmp-version=50 -fnoopenmp-use-tls -ferror-limit 100 -o - %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify=expected,omp45 -fopenmp-version=45 -fopenmp-simd -fnoopenmp-use-tls -ferror-limit 100 -o - %s

#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

int a, b, z; // omp5-error {{variable captured in declare target region must appear in a to clause}}
__thread int t; // expected-note {{defined as threadprivate or thread local}}

#pragma omp declare target . // expected-error {{expected '(' after 'declare target'}}

#pragma omp declare target
void f();
#pragma omp end declare target shared(a) // expected-warning {{extra tokens at the end of '#pragma omp end declare target' are ignored}}

#pragma omp declare target map(a) // omp45-error {{unexpected 'map' clause, only 'to' or 'link' clauses expected}} omp5-error {{unexpected 'map' clause, only 'to', 'link' or 'device_type' clauses expected}}

#pragma omp declare target to(foo1) // expected-error {{use of undeclared identifier 'foo1'}}

#pragma omp declare target link(foo2) // expected-error {{use of undeclared identifier 'foo2'}}

#pragma omp declare target to(f) device_type(any) device_type(any) device_type(host) // omp45-error {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}} omp5-warning 2 {{more than one 'device_type' clause is specified}} omp5-error {{'device_type(host)' does not match previously specified 'device_type(any)' for the same declaration}}

void c();

void func() {} // expected-note {{'func' defined here}}

#pragma omp declare target link(func) allocate(a) // expected-error {{function name is not allowed in 'link' clause}} omp45-error {{unexpected 'allocate' clause, only 'to' or 'link' clauses expected}} omp5-error {{unexpected 'allocate' clause, only 'to', 'link' or 'device_type' clauses expected}}

void bar();
void baz() {bar();}
#pragma omp declare target(bar) // omp5-warning {{declaration marked as declare target after first use, it may lead to incorrect results}}

extern int b;

struct NonT {
  int a;
};

typedef int sint;

template <typename T>
T bla1() { return 0; }

#pragma omp declare target
template <typename T>
T bla2() { return 0; }
#pragma omp end declare target

template<>
float bla2() { return 1.0; }

#pragma omp declare target
void blub2() {
  bla2<float>();
  bla2<int>();
}
#pragma omp end declare target

void t2() {
#pragma omp target
  {
    bla2<float>();
    bla2<long>();
  }
}

#pragma omp declare target
  void abc();
#pragma omp end declare target
void cba();
#pragma omp end declare target // expected-error {{unexpected OpenMP directive '#pragma omp end declare target'}}

#pragma omp declare target
#pragma omp declare target
void def();
#pragma omp end declare target
void fed();

#pragma omp declare target
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

void foo(int p) {
  a = 0; // expected-error {{threadprivate variables cannot be used in target constructs}}
  b = 0;
  t = 1; // expected-error {{threadprivate variables cannot be used in target constructs}}
  C object;
  VC object1;
  g = object.method();
  g += object.method1();
  g += object1.method() + p;
  f();
  c();
}
#pragma omp declare target
void foo1() {
  [&](){ (void)(b+z);}(); // omp5-note {{variable 'z' is captured here}}
}
#pragma omp end declare target

#pragma omp end declare target
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
  foo(v);
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

void bazz() {}
#pragma omp declare target to(bazz) device_type(nohost) // omp45-error {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}} host5-note 3{{marked as 'device_type(nohost)' here}}
void bazzz() {bazz();}
#pragma omp declare target to(bazzz) device_type(nohost) // omp45-error {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
void any() {bazz();} // host5-error {{function with 'device_type(nohost)' is not available on host}}
void host1() {bazz();} // host5-error {{function with 'device_type(nohost)' is not available on host}}
#pragma omp declare target to(host1) device_type(host) // omp45-error {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}} dev5-note 4 {{marked as 'device_type(host)' here}}
void host2() {bazz();} //host5-error {{function with 'device_type(nohost)' is not available on host}}
#pragma omp declare target to(host2)
void device() {host1();} // dev5-error {{function with 'device_type(host)' is not available on device}}
#pragma omp declare target to(device) device_type(nohost) // omp45-error {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}} host5-note 2 {{marked as 'device_type(nohost)' here}}
void host3() {host1();} // dev5-error {{function with 'device_type(host)' is not available on device}}
#pragma omp declare target to(host3)

#pragma omp declare target
void any1() {any();}
void any2() {host1();} // dev5-error {{function with 'device_type(host)' is not available on device}}
void any3() {device();} // host5-error {{function with 'device_type(nohost)' is not available on host}}
void any4() {any2();}
#pragma omp end declare target

void any5() {any();}
void any6() {host1();} // dev5-error {{function with 'device_type(host)' is not available on device}}
void any7() {device();} // host5-error {{function with 'device_type(nohost)' is not available on host}}
void any8() {any2();}

#pragma omp declare target // expected-error {{expected '#pragma omp end declare target'}} expected-note {{to match this '#pragma omp declare target'}}
