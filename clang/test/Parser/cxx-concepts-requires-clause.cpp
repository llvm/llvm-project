// RUN: %clang_cc1 -std=c++20 -x c++ %s -verify
// RUN: %clang_cc1 -std=c++23 -x c++ %s -verify

// Test parsing of the optional requires-clause in a template-declaration.

template <typename T> requires true
void foo() { }

template <typename T> requires (!0)
struct A {
  void foo();
  struct AA;
  enum E : int;
  static int x;
  static constexpr int z = 16;

  template <typename> requires true
  void Mfoo();

  template <typename> requires true
  struct M;

  template <typename> requires true
  static int Mx;

  template <typename TT> requires true
  using MQ = M<TT>;

  constexpr int bazz() requires (z == 16);
};

template <typename T> requires (!0)
void A<T>::foo() { }

template <typename T> requires (!0)
struct A<T>::AA { };

template <typename T> requires (!0)
enum A<T>::E : int { E0 };

template <typename T> requires (!0)
int A<T>::x = 0;

template <typename T> requires (!0)
template <typename> requires true
void A<T>::Mfoo() { }

template <typename T> requires (!0)
template <typename> requires true
struct A<T>::M { };

template <typename T> requires (!0)
template <typename> requires true
int A<T>::Mx = 0;

template <typename T> requires true
int x = 0;

template <typename T> requires true
using Q = A<T>;

template<typename T> requires (!0)
constexpr int A<T>::bazz() requires (z == 16) { return z; }

struct C {
  template <typename> requires true
  void Mfoo();

  template <typename> requires true
  struct M;

  template <typename> requires true
  static int Mx;

  template <typename T> requires true
  using MQ = M<T>;
};

template <typename> requires true
void C::Mfoo() { }

template <typename> requires true
struct C::M { };

template <typename> requires true
int C::Mx = 0;

// Test behavior with non-primary-expression requires clauses

template<typename T> requires foo<T>()
// expected-error@-1{{parentheses are required around this expression in a requires clause}}
struct B1 { };

int func() { }

template<typename T> requires func()
// expected-error@-1{{atomic constraint must be of type 'bool' (found '<overloaded function type>')}}
// expected-note@-2{{parentheses are required around this expression in a requires clause}}
struct B2 { };

template<typename T> requires (foo<T>())
struct B3 { };

template<typename T> requires T{}
// expected-error@-1{{parentheses are required around this expression in a requires clause}}
struct B4 { };

template<typename T> requires sizeof(T) == 0
// expected-error@-1{{parentheses are required around this expression in a requires clause}}
struct B5 { };

template<typename T> requires (sizeof(T)) == 0
// expected-error@-1{{parentheses are required around this expression in a requires clause}}
struct B6 { };

template<typename T> requires 0
// expected-error@-1{{atomic constraint must be of type 'bool' (found 'int')}}
(int) bar() { };

template<typename T> requires foo<T>
(int) bar() { };
// expected-error@-1{{expected '(' for function-style cast or type construction}}

template<typename T>
void bar() requires foo<T>();
// expected-error@-1{{parentheses are required around this expression in a requires clause}}

template<typename T>
void bar() requires (foo<T>());

template<typename T>
void bar() requires func();
// expected-error@-1{{atomic constraint must be of type 'bool' (found '<overloaded function type>')}}
// expected-note@-2{{parentheses are required around this expression in a requires clause}}

template<typename T>
void bar() requires T{};
// expected-error@-1{{parentheses are required around this expression in a requires clause}}

template<typename T>
void bar() requires sizeof(T) == 0;
// expected-error@-1{{parentheses are required around this expression in a requires clause}}

template<typename T>
void bar() requires (sizeof(T)) == 0;
// expected-error@-1{{parentheses are required around this expression in a requires clause}}

template<typename T>
void bar(int x, int y) requires (x, y, true);

template<typename T>
struct B {
  int x;
  void foo(int y) requires (x, this, this->x, y, true);
  static void bar(int y) requires (x, true);
  // expected-error@-1{{'this' cannot be implicitly used in a static member function declaration}}
  static void baz(int y) requires (this, true);
  // expected-error@-1{{'this' cannot be used in a static member function declaration}}
};

auto lambda1 = [] (auto x) requires (sizeof(decltype(x)) == 1) { };

auto lambda2 = [] (auto x) constexpr -> int requires (sizeof(decltype(x)) == 1) { return 0; };

auto lambda3 = []<auto> requires(sizeof(char) == 1){};

auto lambda4 = [] requires(sizeof(char) == 1){}; // expected-error {{expected body of lambda expression}}
#if __cplusplus <= 202002L
// expected-warning@-2{{lambda without a parameter clause is a C++23 extension}}
#endif
