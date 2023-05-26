// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace GH62361 {
  template <typename T, typename U = void*> struct B { // expected-note 14{{candidate}}
    B() // expected-note 7{{not viable}}
      requires __is_same(T, int); // expected-note 7{{because '__is_same(char, int)' evaluated to false}}
  };

  template <typename U> struct B<void, U> : B<int, U> {
    using B<int, U>::B;
  };

  template<typename T>
  void g(B<T>); // expected-note {{cannot convert}}

  void f1() {
    B<void> b1;
    B<void> b2{};
    B<void> b3 = {};
    new B<void>{};
    new B<void>();
    g<void>({});
    B<void>{};
    B<void>();
  }

  void f2() {
    B<int> b1;
    B<int> b2{};
    B<int> b3 = {};
    new B<int>{};
    new B<int>();
    g<int>({});
    B<int>{};
    B<int>();
  }

  void f3() {
    B<char> b1; // expected-error {{no matching constructor}}
    B<char> b2{}; // expected-error {{no matching constructor}}
    B<char> b3 = {}; // expected-error {{no matching constructor}}
    new B<char>{}; // expected-error {{no matching constructor}}
    new B<char>(); // expected-error {{no matching constructor}}
    g<char>({}); // expected-error {{no matching function}}
    B<char>{}; // expected-error {{no matching constructor}}
    B<char>(); // expected-error {{no matching constructor}}
  }
}

namespace no_early_substitution {
  template <typename T> concept X = true;

  struct A {};

  template <typename T> struct B {
    B() requires X<T*>;
    B();
  };

  template <typename U = int, typename V = A>
  struct C : public B<V&> {
    using B<V&>::B;
  };

  void foo() {
    // OK, we only substitute T ~> V& into X<T*> in a SFINAE context,
    // during satisfaction checks.
    C();
  }
}

namespace GH62362 {
  template<typename T>
    concept C = true;
  template <typename T> struct Test {
    Test()
      requires(C<T>);
  };
  struct Bar : public Test<int> {
    using Test<int>::Test;
  };
  template <>
    struct Test<void> : public Test<int> {
      using Test<int>::Test;
    };

  void foo() {
    Bar();
    Test<void>();
  }
}
