// RUN: %clang_cc1 -triple x86_64 -verify=expected,dev -std=c++11\
// RUN:            -verify-ignore-unexpected=note \
// RUN:            -fopenmp -fopenmp-version=50 -o - %s

// expected-no-diagnostics

// Test no infinite recursion in DeferredDiagnosticEmitter.
constexpr int foo(int *x) {
  return 0;
}

int a = foo(&a);

// Test no crash when both caller and callee have target directives.
int foo();

class B {
public:
  void barB(int *isHost) {
  #pragma omp target map(tofrom: isHost)
     {
       *isHost = foo();
     }
  }
};

class A : public B {
public:
  void barA(int *isHost) {
  #pragma omp target map(tofrom: isHost)
     {
       barB(isHost);
     }
  }
};
