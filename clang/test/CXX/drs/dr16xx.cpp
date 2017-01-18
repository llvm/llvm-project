// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++1z -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

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

namespace dr1638 { // dr1638: yes
#if __cplusplus >= 201103L
  template<typename T> struct A {
    enum class E; // expected-note {{previous}}
    enum class F : T; // expected-note 2{{previous}}
  };

  template<> enum class A<int>::E;
  template<> enum class A<int>::E {};
  template<> enum class A<int>::F : int;
  template<> enum class A<int>::F : int {};

  template<> enum class A<short>::E : int;
  template<> enum class A<short>::E : int {};

  template<> enum class A<short>::F; // expected-error {{different underlying type}}
  template<> enum class A<char>::E : char; // expected-error {{different underlying type}}
  template<> enum class A<char>::F : int; // expected-error {{different underlying type}}

  enum class A<unsigned>::E; // expected-error {{template specialization requires 'template<>'}} expected-error {{nested name specifier}}
  template enum class A<unsigned>::E; // expected-error {{enumerations cannot be explicitly instantiated}}
  enum class A<unsigned>::E *e; // expected-error {{must use 'enum' not 'enum class'}}

  struct B {
    friend enum class A<unsigned>::E; // expected-error {{must use 'enum' not 'enum class'}}
  };
#endif
}

namespace dr1645 { // dr1645: 3.9
#if __cplusplus >= 201103L
  struct A {
    constexpr A(int, float = 0); // expected-note 2{{candidate}}
    explicit A(int, int = 0); // expected-note 2{{candidate}}
    A(int, int, int = 0) = delete; // expected-note {{candidate}}
  };

  struct B : A { // expected-note 2{{candidate}}
    using A::A; // expected-note 5{{inherited here}}
  };

  constexpr B a(0); // expected-error {{ambiguous}}
  constexpr B b(0, 0); // expected-error {{ambiguous}}
#endif
}

namespace dr1653 { // dr1653: 4 c++17
  void f(bool b) {
    ++b;
    b++;
#if __cplusplus <= 201402L
    // expected-warning@-3 {{deprecated}} expected-warning@-2 {{deprecated}}
#else
    // expected-error@-5 {{incrementing expression of type bool}} expected-error@-4 {{incrementing expression of type bool}}
#endif
    --b; // expected-error {{cannot decrement expression of type bool}}
    b--; // expected-error {{cannot decrement expression of type bool}}
    b += 1; // ok
    b -= 1; // ok
  }
}
