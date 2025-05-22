// RUN: %clang_cc1 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter

/// This is like the version in test/SemaCXX/, but some of the
/// output types and their location has been adapted.
/// Differences:
///   1) The type of the uninitialized base class is printed WITH the namespace,
///      i.e. 'baseclass_uninit::DelBase' instead of just 'DelBase'.
///   2) The location is not the base specifier declaration, but the call site
///      of the constructor.


namespace baseclass_uninit {
struct DelBase {
  constexpr DelBase() = delete; // expected-note {{'DelBase' has been explicitly marked deleted here}}
};

struct Foo : DelBase {
  constexpr Foo() {}; // expected-error {{call to deleted constructor of 'DelBase'}}
};
constexpr Foo f; // expected-error {{must be initialized by a constant expression}} \
                 // expected-note {{constructor of base class 'baseclass_uninit::DelBase' is not called}}

struct Bar : Foo {
  constexpr Bar() {};
};
constexpr Bar bar; // expected-error {{must be initialized by a constant expression}} \
                   // expected-note {{constructor of base class 'baseclass_uninit::DelBase' is not called}}

struct Base {};
struct A : Base {
  constexpr A() : value() {} // expected-error {{member initializer 'value' does not name a non-static data member or base class}}
};

constexpr A a; // expected-error {{must be initialized by a constant expression}} \
               // expected-note {{constructor of base class 'baseclass_uninit::Base' is not called}}


struct B : Base {
  constexpr B() : {} // expected-error {{expected class member or base class name}}
};

constexpr B b; // expected-error {{must be initialized by a constant expression}} \
               // expected-note {{constructor of base class 'baseclass_uninit::Base' is not called}}
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
