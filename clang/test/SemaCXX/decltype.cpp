// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -Wno-c99-designator %s

// PR5290
int const f0();
void f0_test() {
  decltype(0, f0()) i = 0;
  i = 0;
}

struct A { int a[1]; A() { } };
typedef A const AC;
int &f1(int*);
float &f2(int const*);

void test_f2() {
  float &fr = f2(AC().a);
}

template <class T>
struct Future {
  explicit Future(T v);

  template <class F>
  auto call(F&& fn) -> decltype(fn(T())) {
    return fn(T());
  }

  template <class B, class F>
  auto then(F&& fn) -> decltype(call(fn))
  {
    return fn(T());
  }
};

void rdar16527205() {
  Future<int> f1(42);
  f1.call([](int){ return Future<float>(0); });
}

namespace pr10154 {
  class A{
      A(decltype(nullptr) param);
  };
}

template<typename T> struct S {};
template<typename T> auto f(T t) -> decltype(S<int>(t)) {
  using U = decltype(S<int>(t));
  using U = S<int>;
  return S<int>(t);
}

struct B {
  B(decltype(undeclared)); // expected-error {{undeclared identifier}}
};
struct C {
  C(decltype(undeclared; // expected-error {{undeclared identifier}} \
                         // expected-error {{expected ')'}} expected-note {{to match this '('}}
};

namespace PR16529 {
  struct U {};
  template <typename T> struct S {
    static decltype(T{}, U{}) &f();
  };
  U &r = S<int>::f();
}

namespace PR18876 {
  struct A { ~A() = delete; }; // expected-note +{{here}}
  A f();
  decltype(f()) *a; // ok, function call
  decltype(A()) *b; // expected-error {{attempt to use a deleted function}}
  decltype(0, f()) *c; // ok, function call on RHS of comma
  decltype(0, A()) *d; // expected-error {{attempt to use a deleted function}}
  decltype(f(), 0) *e; // expected-error {{attempt to use a deleted function}}
}

namespace D5789 {
  struct P1 { char x[6]; } g1 = { "foo" };
  struct LP1 { struct P1 p1; };

  // expected-warning@+3 {{initializer partially overrides}}
  // expected-note@+2 {{previous initialization}}
  // expected-note@+1 {{previous definition}}
  template<class T> void foo(decltype(T(LP1{ .p1 = g1, .p1.x[1] = 'x' }))) {}

  // expected-warning@+3 {{initializer partially overrides}}
  // expected-note@+2 {{previous initialization}}
  template<class T>
  void foo(decltype(T(LP1{ .p1 = g1, .p1.x[1] = 'r' }))) {} // okay

  // expected-warning@+3 {{initializer partially overrides}}
  // expected-note@+2 {{previous initialization}}
  template<class T>
  void foo(decltype(T(LP1{ .p1 = { "foo" }, .p1.x[1] = 'x'}))) {} // okay

  // expected-warning@+3 {{initializer partially overrides}}
  // expected-note@+2 {{previous initialization}}
  // expected-error@+1 {{redefinition of 'foo'}}
  template<class T> void foo(decltype(T(LP1{ .p1 = g1, .p1.x[1] = 'x' }))) {}
}

namespace GH58674 {
  struct Foo {
    float value_;
    struct nested {
      float value_;
    };
  };

  template <typename T>
  struct TemplateFoo {
    float value_;
  };

  float bar;

  template <typename T>
  struct Animal{};

  template <typename T>
  class Cat : Animal<T> {
    using okay = decltype(Foo::value_);
    using also_okay = decltype(bar);
    using okay2 = decltype(Foo::nested::value_);
    using okay3 = decltype(TemplateFoo<T>::value_);
  public:
    void meow() {
      using okay = decltype(Foo::value_);
      using also_okay = decltype(bar);
      using okay2 = decltype(Foo::nested::value_);
      using okay3 = decltype(TemplateFoo<T>::value_);
    }
  };

  void baz() {
      Cat<void>{}.meow();
  }
}

template<typename>
class conditional {
};

// FIXME: The diagnostics here are produced twice.
void foo(conditional<decltype((1),int>) {  // expected-note 2 {{to match this '('}} expected-error {{expected ')'}} expected-note 2{{to match this '<'}}
} // expected-error {{expected function body after function declarator}} expected-error 2 {{expected '>'}} expected-error {{expected ')'}}
