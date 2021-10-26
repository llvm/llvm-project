// RUN: %clang_cc1 -triple x86_64 -verify=expected,dev -std=c++11\
// RUN:            -verify-ignore-unexpected=note \
// RUN:            -fopenmp -fopenmp-version=45 -o - %s

// RUN: %clang_cc1 -triple x86_64 -verify=expected,dev -std=c++11\
// RUN:            -verify-ignore-unexpected=note \
// RUN:            -fopenmp -o - %s

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

// Test that deleting an incomplete class type doesn't cause an assertion.
namespace TestDeleteIncompleteClassDefinition {
struct a;
struct b {
  b() {
    delete c; // expected-warning {{deleting pointer to incomplete type 'TestDeleteIncompleteClassDefinition::a' may cause undefined behavior}}
  }
  a *c;
};
} // namespace TestDeleteIncompleteClassDefinition
