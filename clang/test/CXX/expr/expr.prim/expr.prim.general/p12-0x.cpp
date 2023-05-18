// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  int *j = &nonexistent; // expected-error {{use of undeclared identifier 'nonexistent'}}
  int *m = &n; // ok

  int n = f(); // ok
  int f();
};

int i = sizeof(S::m); // ok
int j = sizeof(S::m + 42); // ok


struct T {
  int n;
  static void f() {
    int a[n]; // expected-error {{invalid use of member 'n' in static member function}}
    int b[sizeof n]; // ok
  }
};

// Make sure the rule for unevaluated operands works correctly with typeid.
namespace std {
  class type_info;
}
class Poly { virtual ~Poly(); };
const std::type_info& k = typeid(S::m);
const std::type_info& m = typeid(*(Poly*)S::m); // expected-error {{invalid use of non-static data member}}
const std::type_info& n = typeid(*(Poly*)(0*sizeof S::m)); 

namespace PR11956 {
  struct X { char a; };
  struct Y { int f() { return sizeof(X::a); } }; // ok

  struct A { enum E {} E; };
  struct B { int f() { return sizeof(A::E); } }; // ok
}
