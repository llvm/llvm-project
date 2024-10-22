// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2c %s

struct A;
struct B;
struct C;

struct S {};
template <typename> struct TS {};

template <typename ...Pack>
class X {
  friend Pack...;
  static void f() { } // expected-note {{declared private here}}
};

class Y {
  friend A, B, C;
  static void g() { } // expected-note {{declared private here}}
};

struct A {
  A() {
    X<A>::f();
    Y::g();
  };
};

struct B {
  B() {
    X<B, C>::f();
    Y::g();
  };
};

struct C {
  C() {
    X<A, B, C>::f();
    Y::g();
  };
};

struct D {
  D() {
    X<A, B, C>::f(); // expected-error {{'f' is a private member of 'X<A, B, C>'}}
    Y::g(); // expected-error {{'g' is a private member of 'Y'}}
  };
};

void f1() {
  A a;
  B b;
  C c;
  D d;
}

template <typename ...Pack>
struct Z {
  template <template <typename> class Template>
  struct Inner {
    friend Template<Pack>...;
  };
};

void f2() {
  Z<int, long, char> z;
  Z<int, long, char>::Inner<TS> inner;
}

namespace p2893r3_examples {
template<class... Ts>
class Passkey {
  friend Ts...;
  Passkey() {} // expected-note {{declared private here}}
};

class Foo;
class Bar;
class Baz;

class C {
public:
  void f(Passkey<Foo, Bar, Baz>);
};

class Foo {
  Foo() { C c; c.f({}); }
};

class Bar {
  Bar() { C c; c.f({}); }
};

class Baz {
  Baz() { C c; c.f({}); }
};

class Quux {
  Quux() { C c; c.f({}); } // expected-error {{calling a private constructor of class 'p2893r3_examples::Passkey<p2893r3_examples::Foo, p2893r3_examples::Bar, p2893r3_examples::Baz>'}}
};

template<class Derived, class MsgT>
struct Receiver {
  void receive(MsgT) {
    static_cast<Derived*>(this)->private_ += 1;
  }
};

template<class... MsgTs>
struct Dispatcher : Receiver<Dispatcher<MsgTs...>, MsgTs>... {
  using Receiver<Dispatcher, MsgTs>::receive...;
  friend Receiver<Dispatcher, MsgTs>...;

private:
  int private_;
};

void f() {
  Dispatcher<int, float> d;
  d.receive(0);
  d.receive(0.0f);
}
} // namespace p2893r3_examples

namespace p2893r3_note {
template <class... Ts> class R {
  friend Ts...;
};

template <class... Ts, class... Us>
class R<R<Ts...>, R<Us...>> {
  friend Ts::Nested..., Us...;
};

struct E { struct Nested; };
R<R<E>, R<C, int>> rr;
} // namespace p2893r3_note

namespace template_template {
template <typename U, template <typename> typename... Friend>
class S {
  friend class Friend<U>...;
  static constexpr int a = 42;
};

template <typename U>
struct T {
  static_assert(S<U, T>::a == 42);
  static_assert(S<U, T>::a == 43); // expected-error {{static assertion failed due to requirement 'S<int, template_template::T>::a == 43'}} \
                                   // expected-note {{expression evaluates to '42 == 43'}}
};

void f() {
  T<int> t; // expected-note {{in instantiation of}}
}
}

