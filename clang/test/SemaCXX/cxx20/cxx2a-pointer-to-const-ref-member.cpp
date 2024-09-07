// RUN: %clang_cc1 -std=c++2a %s -verify

struct X {
  void ref() & {} // expected-note{{'ref' declared here}}
  void cref() const& {}
  void cvref() const volatile & {} // expected-note{{'cvref' declared here}}
};

void test() {
  X{}.ref(); // expected-error{{'this' argument to member function 'ref' is an rvalue, but function has non-const lvalue ref-qualifier}}
  X{}.cref(); // expected-no-error
  X{}.cvref(); // expected-error{{'this' argument to member function 'cvref' is an rvalue, but function has non-const lvalue ref-qualifier}}

  (X{}.*&X::ref)(); // expected-error-re{{pointer-to-member function type 'void (X::*)() {{.*}}&' can only be called on an lvalue}}
  (X{}.*&X::cref)(); // expected-no-error
  (X{}.*&X::cvref)(); // expected-error-re{{pointer-to-member function type 'void (X::*)() {{.*}}&' can only be called on an lvalue}}
}
