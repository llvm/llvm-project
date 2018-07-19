// RUN: %clang_cc1 -std=c++2a %s -verify

struct X {
  void ref() & {} // expected-note{{'ref' declared here}}
  void cref() const& {}
};

void test() {
  X{}.ref(); // expected-error{{'this' argument to member function 'ref' is an rvalue, but function has non-const lvalue ref-qualifier}}
  X{}.cref(); // expected-no-error

  (X{}.*&X::ref)(); // expected-error-re{{pointer-to-member function type 'void (X::*)() {{.*}}&' can only be called on an lvalue}}
  (X{}.*&X::cref)(); // expected-no-error
}
