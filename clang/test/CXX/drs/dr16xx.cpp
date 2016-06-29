// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

#if __cplusplus < 201103L
// expected-no-diagnostics
#endif

namespace dr1684 { // dr1684: 3.6
#if __cplusplus >= 201103L
  struct NonLiteral { // expected-note {{because}}
    NonLiteral();
    constexpr int f() { return 0; } // expected-warning 0-1{{will not be implicitly 'const'}}
  };
  constexpr int f(NonLiteral &) { return 0; }
  constexpr int f(NonLiteral) { return 0; } // expected-error {{not a literal type}}
#endif
}

namespace dr1631 {  // dr1631: 3.7
#if __cplusplus >= 201103L
  // Incorrect overload resolution for single-element initializer-list

  struct A { int a[1]; };
  struct B { B(int); };
  void f(B, int);
  void f(B, int, int = 0);
  void f(int, A);

  void test() {
    f({0}, {{1}}); // expected-warning {{braces around scalar init}}
  }

  namespace with_error {
    void f(B, int);           // TODO: expected- note {{candidate function}}
    void f(int, A);           // expected-note {{candidate function}}
    void f(int, A, int = 0);  // expected-note {{candidate function}}
    
    void test() {
      f({0}, {{1}});        // expected-error{{call to 'f' is ambiguous}}
    }
  }
#endif
}

namespace dr1645 { // dr1645: 3.9
#if __cplusplus >= 201103L
  struct A { // expected-note 2{{candidate}}
    constexpr A(int, float = 0); // expected-note 2{{candidate}}
    explicit A(int, int = 0); // expected-note 2{{candidate}}
    A(int, int, int = 0) = delete; // expected-note {{candidate}}
  };

  struct B : A { // expected-note 2{{candidate}}
    using A::A; // expected-note 7{{inherited here}}
  };

  constexpr B a(0); // expected-error {{ambiguous}}
  constexpr B b(0, 0); // expected-error {{ambiguous}}
#endif
}
