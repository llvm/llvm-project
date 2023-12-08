// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace baseclass_uninit {
struct DelBase {
  constexpr DelBase() = delete; // expected-note {{'DelBase' has been explicitly marked deleted here}}
};

struct Foo : DelBase {  // expected-note 2{{constructor of base class 'DelBase' is not called}}
  constexpr Foo() {}; // expected-error {{call to deleted constructor of 'DelBase'}}
};
constexpr Foo f; // expected-error {{must be initialized by a constant expression}}
struct Bar : Foo {
  constexpr Bar() {};
};
constexpr Bar bar; // expected-error {{must be initialized by a constant expression}}

struct Base {};
struct A : Base { // expected-note {{constructor of base class 'Base' is not called}}
  constexpr A() : value() {} // expected-error {{member initializer 'value' does not name a non-static data member or base class}}
};

constexpr A a; // expected-error {{must be initialized by a constant expression}}

struct B : Base { // expected-note {{constructor of base class 'Base' is not called}}
  constexpr B() : {} // expected-error {{expected class member or base class name}}
};

constexpr B b; // expected-error {{must be initialized by a constant expression}}
} // namespace baseclass_uninit


struct Foo {
  constexpr Foo(); // expected-note 2{{declared here}}
};

constexpr Foo ff; // expected-error {{must be initialized by a constant expression}} \
                  // expected-note {{undefined constructor 'Foo' cannot be used in a constant expression}}

struct Bar : protected Foo {
  int i;
  constexpr Bar() : i(12) {} // expected-note {{undefined constructor 'Foo' cannot be used in a constant expression}}
};

constexpr Bar bb; // expected-error {{must be initialized by a constant expression}} \
                  // expected-note {{in call to 'Bar()'}}

template <typename Ty>
struct Baz {
  constexpr Baz(); // expected-note {{declared here}}
};

struct Quux : Baz<Foo>, private Bar {
  int i;
  constexpr Quux() : i(12) {} // expected-note {{undefined constructor 'Baz' cannot be used in a constant expression}}
};

constexpr Quux qx; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{in call to 'Quux()'}}
