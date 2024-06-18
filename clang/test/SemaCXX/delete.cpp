// Test without PCH
// RUN: %clang_cc1 -fsyntax-only -include %S/delete-mismatch.h -fdiagnostics-parseable-fixits -std=c++11 %s 2>&1 | FileCheck %s

// Test with PCH
// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -o %t %S/delete-mismatch.h
// RUN: %clang_cc1 -std=c++11 -include-pch %t -DWITH_PCH -verify %s -ast-dump

void f(int a[10][20]) {
  delete a; // expected-warning {{'delete' applied to a pointer-to-array type}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:9}:"[]"
}
namespace MemberCheck {
struct S {
  int *a = new int[5]; // expected-note4 {{allocated with 'new[]' here}}
  int *b;
  int *c;
  static int *d;
  S();
  S(int);
  ~S() {
    delete a; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
    delete b;   // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
    delete[] c; // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}
  }
  void f();
};

void S::f()
{
  delete a; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  delete b; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
}

S::S()
: b(new int[1]), c(new int) {} // expected-note3 {{allocated with 'new[]' here}}
// expected-note@-1 {{allocated with 'new' here}}

S::S(int i)
: b(new int[i]), c(new int) {} // expected-note3 {{allocated with 'new[]' here}}
// expected-note@-1 {{allocated with 'new' here}}

struct S2 : S {
  ~S2() {
    delete a; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  }
};
int *S::d = new int[42]; // expected-note {{allocated with 'new[]' here}}
void f(S *s) {
  int *a = new int[1]; // expected-note {{allocated with 'new[]' here}}
  delete a; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  delete s->a; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  delete s->b; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  delete s->c;
  delete s->d;
  delete S::d; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
}

// At least one constructor initializes field with matching form of 'new'.
struct MatchingNewIsOK {
  int *p;
  bool is_array_;
  MatchingNewIsOK() : p{new int}, is_array_(false) {}
  explicit MatchingNewIsOK(unsigned c) : p{new int[c]}, is_array_(true) {}
  ~MatchingNewIsOK() {
    if (is_array_)
      delete[] p;
    else
      delete p;
  }
};

// At least one constructor's body is missing; no proof of mismatch.
struct CantProve_MissingCtorDefinition {
  int *p;
  CantProve_MissingCtorDefinition();
  CantProve_MissingCtorDefinition(int);
  ~CantProve_MissingCtorDefinition();
};

CantProve_MissingCtorDefinition::CantProve_MissingCtorDefinition()
  : p(new int)
{ }

CantProve_MissingCtorDefinition::~CantProve_MissingCtorDefinition()
{
  delete[] p;
}

struct base {};
struct derived : base {};
struct InitList {
  base *p, *p2 = nullptr, *p3{nullptr}, *p4;
  InitList(unsigned c) : p(new derived[c]), p4(nullptr) {}  // expected-note {{allocated with 'new[]' here}}
  InitList(unsigned c, unsigned) : p{new derived[c]}, p4{nullptr} {} // expected-note {{allocated with 'new[]' here}}
  ~InitList() {
    delete p; // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
    delete [] p;
    delete p2;
    delete [] p3;
    delete p4;
  }
};
}

namespace NonMemberCheck {
#define DELETE_ARRAY(x) delete[] (x)
#define DELETE(x) delete (x)
void f() {
  int *a = new int(5); // expected-note2 {{allocated with 'new' here}}
  delete[] a;          // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}
  int *b = new int;
  delete b;
  int *c{new int};    // expected-note {{allocated with 'new' here}}
  int *d{new int[1]}; // expected-note2 {{allocated with 'new[]' here}}
  delete  [    ] c;   // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:17}:""
  delete d;           // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:9-[[@LINE-1]]:9}:"[]"
  DELETE_ARRAY(a);    // expected-warning {{'delete[]' applied to a pointer that was allocated with 'new'; did you mean 'delete'?}}
  DELETE(d);          // expected-warning {{'delete' applied to a pointer that was allocated with 'new[]'; did you mean 'delete[]'?}}
}
}

namespace MissingInitializer {
template<typename T>
struct Base {
  struct S {
    const T *p1 = nullptr;
    const T *p2 = new T[3];
  };
};

void null_init(Base<double>::S s) {
  delete s.p1;
  delete s.p2;
}
}

#ifndef WITH_PCH
pch_test::X::X()
  : a(new int[1])  // expected-note{{allocated with 'new[]' here}}
{ }
pch_test::X::X(int i)
  : a(new int[i])  // expected-note{{allocated with 'new[]' here}}
{ }
#endif
