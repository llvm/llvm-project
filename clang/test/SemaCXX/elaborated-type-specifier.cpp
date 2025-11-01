// RUN: %clang_cc1 -fsyntax-only -verify %s

// Test the use of elaborated-type-specifiers to inject the names of
// structs (or classes or unions) into an outer scope as described in
// C++ [basic.scope.pdecl]p5.
typedef struct S1 {
  union {
    struct S2 *x;
    struct S3 *y;
  };
} S1;

bool test_elab(S1 *s1, struct S2 *s2, struct S3 *s3) {
  if (s1->x == s2) return true;
  if (s1->y == s3) return true;
  return false;
}

namespace NS {
  class X {
  public:
    void test_elab2(struct S4 *s4); // expected-note{{'NS::S4' declared here}}
  };

  void X::test_elab2(S4 *s4) { } // expected-note{{passing argument to parameter 's4' here}}
}

void test_X_elab(NS::X x) {
  struct S4 *s4 = 0; // expected-note{{'S4' is not defined, but forward declared here; conversion would be valid if it was derived from 'NS::S4'}}
  x.test_elab2(s4); // expected-error{{cannot initialize a parameter of type 'S4 *' with an lvalue of type 'struct S4 *'}}
}

namespace NS {
  S4 *get_S4();
}

void test_S5_scope() {
  S4 *s4; // expected-error{{unknown type name 'S4'; did you mean 'NS::S4'?}}
}

int test_funcparam_scope(struct S5 * s5) {
  struct S5 { int y; } *s5_2 = 0;
  if (s5 == s5_2) return 1; // expected-error {{comparison of distinct pointer types ('struct S5 *' and 'struct S5 *')}}
  return 0;
}

namespace test5 {
  struct A {
    class __attribute__((visibility("hidden"))) B {};

    void test(class __attribute__((visibility("hidden"), noreturn)) B b) { // expected-warning {{'noreturn' attribute only applies to functions and methods}}
    }
  };
}

namespace test6 {
struct C {
  template <typename> friend struct A; // expected-note {{'A' declared here}}
};
struct B {
  struct A *p; // expected-error {{implicit declaration introduced by elaborated type conflicts with a template of the same name}}
};
}
