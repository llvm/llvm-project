// DEFINE: %{common_opts_mac} = -triple x86_64-apple-macos10.7.0
// DEFINE: %{limit} = -fnoopenmp-use-tls -ferror-limit 100 
// DEFINE: %{target_mac} = -fopenmp-targets=x86_64-apple-macos10.7.0
// DEFINE: %{aux_triple} = -aux-triple x86_64-apple-macos10.7.0
// DEFINE: %{openmp45} = -fopenmp -fopenmp-version=45
// DEFINE: %{openmp50} = -fopenmp -fopenmp-version=50
// DEFINE: %{openmp50_simd} = -fopenmp-simd -fopenmp-version=50
// DEFINE: %{openmp52} = -fopenmp -fopenmp-version=52
// DEFINE: %{openmp60} = -fopenmp -fopenmp-version=60
// DEFINE: %{openmp60_simd} = -fopenmp-simd -fopenmp-version=60

// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp45,omp45-to-51,omp45-to-51-var,omp45-to-51-clause,omp45-to-51-clause  %{openmp45} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp5,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host5,host-5-and-51,no-host5-and-51  %{openmp50} %{target_mac} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var  %{openmp60} %{target_mac} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp5,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51,dev5  %{openmp50} -fopenmp-is-target-device %{target_mac} %{aux_triple} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp60} -fopenmp-is-target-device %{target_mac} %{aux_triple} %{limit} -o - %s

// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp5,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host5,host-5-and-51,no-host5-and-51 %{openmp50_simd} %{target_mac} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp60_simd} %{target_mac} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp5,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host5,host-5-and-51,no-host5-and-51 %{openmp50_simd} -fopenmp-is-target-device %{target_mac} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp60_simd} -fopenmp-is-target-device %{target_mac} %{limit} -o - %s

// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp45,omp45-to-51,omp45-to-51-var,omp45-to-51-clause -fopenmp-version=45 -fopenmp-simd %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp51,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 -fopenmp %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp51,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 -fopenmp %{limit} -DTESTEND=1 -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp51,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 -fopenmp %{limit} -I%S/Inputs -DTESTENDINC=1 -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp51,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 -fopenmp-simd %{limit} -o - %s

// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp52} -DVERBOSE_MODE=1 %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp60} -DVERBOSE_MODE=1 %{limit} -o - %s

// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp5,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 %{openmp50} %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp51,ompvar,omp45-to-51,omp5-and-51,omp5-or-later,omp5-or-later-var,omp45-to-51-var,omp45-to-51-clause,host-5-and-51,no-host5-and-51 -fopenmp %{limit} -o - %s
// RUN: %clang_cc1 %{common_opts_mac} -verify=expected,omp52,ompvar,omp5-or-later,omp5-or-later-var %{openmp60} %{limit} -o - %s


// expected-error@+1 {{unexpected OpenMP directive '#pragma omp end declare target'}}
#pragma omp end declare target 

// ompvar-error@+1 {{variable captured in declare target region must appear in a to clause}}
int a, b, z;
// expected-note@+1 {{defined as threadprivate or thread local}}
__thread int t;

// expected-error@+1 {{expected '(' after 'declare target'}}
#pragma omp declare target . 

#pragma omp declare target
void f();
// expected-warning@+1 {{extra tokens at the end of '#pragma omp end declare target' are ignored}}
#pragma omp end declare target shared(a) 

// omp52-error@+8 {{unexpected 'map' clause, only 'enter', 'link', 'device_type' or 'indirect' clauses expected}}
// omp52-error@+7 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp51-error@+6 {{unexpected 'map' clause, only 'to', 'link', 'device_type' or 'indirect' clauses expected}} 
// omp51-error@+5 {{expected at least one 'to', 'link' or 'indirect' clause}}
// omp5-error@+4 {{unexpected 'map' clause, only 'to', 'link' or 'device_type' clauses expected}}
// omp5-error@+3 {{expected at least one 'to' or 'link' clause}}
// omp45-error@+2 {{unexpected 'map' clause, only 'to' or 'link' clauses expected}}
// omp45-error@+1 {{expected at least one 'to' or 'link' clause}} 
#pragma omp declare target map(a)

// omp52-error@+3 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp45-to-51-error@+1 {{use of undeclared identifier 'foo1'}}
#pragma omp declare target to(foo1) 

// expected-error@+1 {{use of undeclared identifier 'foo2'}}
#pragma omp declare target link(foo2) 

// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// dev5-note@+2 {{marked as 'device_type(host)' here}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(f) device_type(host)

void q();
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp5-and-51-warning@+2 {{more than one 'device_type' clause is specified}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(q) device_type(any) device_type(any) device_type(host) 

#if _OPENMP == 202011
// omp51-error@+1 {{directive '#pragma omp declare target' cannot contain more than one 'indirect' clause}}
#pragma omp declare target to(q) indirect(true) indirect(false)

// expected-note@+1 {{declared here}}
int xxx;
// omp51-error@+2 {{expression is not an integral constant expression}}
// omp51-note@+1 {{read of non-const variable 'xxx' is not allowed in a constant expression}}
#pragma omp declare target to(q) indirect(xxx)

constexpr bool fz() {return true;}
// omp51-error@+1 {{unexpected 'to' clause, only 'device_type', 'indirect' clauses expected}}
#pragma omp begin declare target to(q) indirect(fz()) device_type(nohost)
#pragma omp end declare target

// omp51-error@+1 {{unexpected 'to' clause, only 'device_type', 'indirect' clauses expected}}
#pragma omp begin declare target indirect to(xxx)
void bar();
#pragma omp end declare target

// omp51-error@+2 {{unexpected 'tofrom' clause, only 'to', 'link', 'device_type' or 'indirect' clauses expected}}
// omp51-error@+1 {{expected at least one 'to', 'link' or 'indirect' clause}}
#pragma omp declare target tofrom(xxx)

// omp51-error@+1 {{only 'device_type(any)' clause is allowed with indirect clause}}
#pragma omp begin declare target device_type(host) indirect
void bar();
#pragma omp end declare target
#endif  // _OPENMP

void c();

// expected-note@+1 {{'func' defined here}}
void func() {} 

// omp52-error@+5 {{unexpected 'allocate' clause, only 'enter', 'link', 'device_type' or 'indirect' clauses expected}}
// omp51-error@+4 {{unexpected 'allocate' clause, only 'to', 'link', 'device_type' or 'indirect' clauses expected}}
// omp5-error@+3 {{unexpected 'allocate' clause, only 'to', 'link' or 'device_type' clauses expected}}
// expected-error@+2 {{function name is not allowed in 'link' clause}}
// omp45-error@+1 {{unexpected 'allocate' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target link(func) allocate(a)

void bar();
void baz() {bar();}
// omp5-or-later-warning@+1 {{declaration marked as declare target after first use, it may lead to incorrect results}}
#pragma omp declare target(bar)

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
// expected-error@+1 {{unexpected OpenMP directive '#pragma omp end declare target'}}
#pragma omp end declare target 

#pragma omp declare target
#pragma omp declare target
void def();
#pragma omp end declare target
void fed();

#pragma omp declare target
// expected-note@+1 {{defined as threadprivate or thread local}}
#pragma omp threadprivate(a) 
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
// expected-error@+1 {{threadprivate variables cannot be used in target constructs}}
  a = 0; 
  b = 0;
// expected-error@+1 {{threadprivate variables cannot be used in target constructs}}
  t = 1; 
  C object;
  VC object1;
  g = object.method();
  g += object.method1();
  g += object1.method() + p;
  // dev5-error@+1 {{function with 'device_type(host)' is not available on device}}
  f(); 
  q();
  c();
}
#pragma omp declare target
void foo1() {
  // omp5-or-later-var-note@+1 {{variable 'z' is captured here}}
  [&](){ (void)(b+z);}(); 
}
#pragma omp end declare target

#pragma omp end declare target
#pragma omp end declare target
// expected-error@+1 {{unexpected OpenMP directive '#pragma omp end declare target'}}
#pragma omp end declare target 

int C::method() {
  return 0;
}

struct S {
#pragma omp declare target
  int v;
#pragma omp end declare target
};

void foo3() {
  return;
}

int *y;
int **w = &y;
int main (int argc, char **argv) {
  int a = 2;
// expected-error@+1 {{unexpected OpenMP directive '#pragma omp declare target'}}
#pragma omp declare target 
  int v;
// expected-error@+1 {{unexpected OpenMP directive '#pragma omp end declare target'}}
#pragma omp end declare target 
  foo(v);

  // omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
  // omp52-error@+1 {{unexpected 'to' clause, use 'enter' instead}}
#pragma omp declare target to(foo3) link(w)
  // omp52-error@+3 {{unexpected 'to' clause, use 'enter' instead}}
  // omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
  // omp45-to-51-var-error@+1 {{local variable 'a' should not be used in 'declare target' directive}}
#pragma omp declare target to(a) 
  return (0);
}

namespace {
#pragma omp declare target
  int x;
}
#pragma omp end declare target

// expected-error@+1 {{'S' used in declare target directive is not a variable or a function name}}
#pragma omp declare target link(S) 

// expected-error@+1 {{'x' appears multiple times in clauses on the same declare target directive}}
#pragma omp declare target (x, x) 
// omp52-error@+3 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp45-to-51-clause-error@+1 {{'x' appears multiple times in clauses on the same declare target directive}}
#pragma omp declare target to(x) to(x)
// expected-error@+1 {{'x' must not appear in both clauses 'to' and 'link'}}
#pragma omp declare target link(x) 

void bazz() {}
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// host5-note@+2 3 {{marked as 'device_type(nohost)' here}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}} 
#pragma omp declare target to(bazz) device_type(nohost)
void bazzz() {bazz();}
// omp52-error@+3 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(bazzz) device_type(nohost) 
// host5-error@+1 {{function with 'device_type(nohost)' is not available on host}}
void any() {bazz();} 
// host5-error@+1 {{function with 'device_type(nohost)' is not available on host}}
void host1() {bazz();}
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// dev5-note@+2 3 {{marked as 'device_type(host)' here}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(host1) device_type(host)
//host5-error@+1 {{function with 'device_type(nohost)' is not available on host}}
void host2() {bazz();}
// omp52-error@+2 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+1 {{expected at least one 'enter', 'link' or 'indirect' clause}}
#pragma omp declare target to(host2) 
// dev5-error@+1 {{function with 'device_type(host)' is not available on device}}
void device() {host1();}
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// host5-note@+2 2 {{marked as 'device_type(nohost)' here}} 
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(device) device_type(nohost)
void host3() {host1();} // dev5-error {{function with 'device_type(host)' is not available on device}}
// omp52-error@+2 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+1 {{expected at least one 'enter', 'link' or 'indirect' clause}}
#pragma omp declare target to(host3)

#pragma omp declare target
void any1() {any();}
// dev5-error@+1 {{function with 'device_type(host)' is not available on device}}
void any2() {host1();} 
// host5-error@+1 {{function with 'device_type(nohost)' is not available on host}}
void any3() {device();}
void any4() {any2();}
#pragma omp end declare target

void any5() {any();}
void any6() {host1();}
// host5-error@+1 {{function with 'device_type(nohost)' is not available on host}}
void any7() {device();}
void any8() {any2();}

int MultiDevTy;
// omp52-error@+3 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+2 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(MultiDevTy) device_type(any)
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// host-5-and-51-error@+2 {{'device_type(host)' does not match previously specified 'device_type(any)' for the same declaration}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(MultiDevTy) device_type(host)
// omp52-error@+4 {{unexpected 'to' clause, use 'enter' instead}}
// omp52-error@+3 {{expected at least one 'enter', 'link' or 'indirect' clause}}
// no-host5-and-51-error@+2 {{'device_type(nohost)' does not match previously specified 'device_type(any)' for the same declaration}}
// omp45-error@+1 {{unexpected 'device_type' clause, only 'to' or 'link' clauses expected}}
#pragma omp declare target to(MultiDevTy) device_type(nohost)

// expected-warning@+1 {{declaration is not declared in any declare target region}}
static int variable = 100; 
static float variable1 = 200;
// expected-warning@+1 {{declaration is not declared in any declare target region}}
static float variable2 = variable1;  

// expected-warning@+1 {{declaration is not declared in any declare target region}}
static int var = 1;  

static int var1 = 10;
static int *var2 = &var1;
// expected-warning@+1 {{declaration is not declared in any declare target region}}
static int **ptr1 = &var2;  

int arr[2] = {1,2};
// expected-warning@+1 {{declaration is not declared in any declare target region}}
int (*arrptr)[2] = &arr;  

class declare{
  public: int x;
          void print();
};
declare obj1;
// expected-warning@+1 {{declaration is not declared in any declare target region}}
declare *obj2 = &obj1;  

struct target{
  int x;
  void print();
};
// expected-warning@+1 {{declaration is not declared in any declare target region}}
static target S;  

#pragma omp declare target
// expected-note@+1 {{used here}}
int target_var = variable;  
// expected-note@+1 {{used here}}
float target_var1 = variable2;  
// expected-note@+1 {{used here}}
int *ptr = &var;  
// expected-note@+1 {{used here}}
int ***ptr2 = &ptr1; 
// expected-note@+1 {{used here}}
int (**ptr3)[2] = &arrptr;
// expected-note@+1 {{used here}}
declare **obj3 = &obj2;
// expected-note@+1 {{used here}}
target *S1 = &S;
#pragma omp end declare target

#if TESTENDINC
#include "unterminated_declare_target_include.h"
#elif TESTEND
// expected-warning@+1 {{expected '#pragma omp end declare target' at end of file to match '#pragma omp declare target'}}
#pragma omp declare target
#else
// expected-warning@+1 {{expected '#pragma omp end declare target' at end of file to match '#pragma omp begin declare target'}}
#pragma omp begin declare target
#endif
