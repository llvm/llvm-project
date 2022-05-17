// RUN:  %clang_cc1 -std=c++2a -verify %s

template<typename T, typename U>
constexpr static bool is_same_v = false;

template<typename T>
constexpr static bool is_same_v<T, T> = true;

namespace templates
{
  template<typename T>
  concept AtLeast1 = sizeof(T) >= 1;

  template<typename T>
  int foo(T t) requires (sizeof(T) == 4) { // expected-note {{candidate function}}
    return 0;
  }

  template<typename T>
  char foo(T t) requires AtLeast1<T> { // expected-note {{candidate function}}
    return 'a';
  }

  template<typename T>
  double foo(T t) requires (AtLeast1<T> && sizeof(T) <= 2) {
    return 'a';
  }

  static_assert(is_same_v<decltype(foo(10)), int>); // expected-error {{call to 'foo' is ambiguous}}
  static_assert(is_same_v<decltype(foo(short(10))), double>);

  template<typename T>
  void bar() requires (sizeof(T) == 1) { }
  // expected-note@-1{{similar constraint expressions not considered equivalent}}
  // expected-note@-2{{candidate function [with T = char]}}

  template<typename T>
  void bar() requires (sizeof(T) == 1 && sizeof(T) >= 0) { }
  // expected-note@-1{{candidate function [with T = char]}}
  // expected-note@-2{{similar constraint expression here}}

  static_assert(is_same_v<decltype(bar<char>()), void>);
  // expected-error@-1{{call to 'bar' is ambiguous}}

  template<typename T>
  constexpr int baz() requires AtLeast1<T> { // expected-note {{candidate function}}
    return 1;
  }

  template<typename T> requires AtLeast1<T>
  constexpr int baz() { // expected-note {{candidate function [with T = int]}}
    return 2;
  }

  static_assert(baz<int>() == 1); // expected-error {{call to 'baz' is ambiguous}}
}

namespace non_template
{
  template<typename T>
  concept AtLeast2 = sizeof(T) >= 2;

  template<typename T>
  concept AtMost8 = sizeof(T) <= 8;

  template<typename T>
  int foo() requires AtLeast2<long> && AtMost8<long> {
    return 0;
  }

  template<typename T>
  double foo() requires AtLeast2<long> {
    return 0.0;
  }

  template<typename T>
  double baz() requires AtLeast2<long> && AtMost8<long> { // expected-note {{candidate function}}
    return 0.0;
  }

  template<typename T>
  int baz() requires AtMost8<long> && AtLeast2<long> { // expected-note {{candidate function}}
    return 0.0;
  }

  template<typename T>
  void bar() requires (sizeof(char[8]) >= 8) { }
  // expected-note@-1 {{candidate function}}
  // expected-note@-2 {{similar constraint expressions not considered equivalent}}

  template<typename T>
  void bar() requires (sizeof(char[8]) >= 8 && sizeof(int) <= 30) { }
  // expected-note@-1 {{candidate function}}
  // expected-note@-2 {{similar constraint expression here}}

  static_assert(is_same_v<decltype(foo<int>()), int>);
  static_assert(is_same_v<decltype(baz<int>()), int>); // expected-error {{call to 'baz' is ambiguous}}
  static_assert(is_same_v<decltype(bar<int>()), void>); // expected-error {{call to 'bar' is ambiguous}}

  template<typename T>
  constexpr int goo(int a) requires AtLeast2<int> && true {
    return 1;
  }

  template<typename T>
  constexpr int goo(const int b) requires AtLeast2<int> {
    return 2;
  }

  // Only trailing requires clauses of redeclarations are compared for overload resolution.
  template<typename T>
  constexpr int doo(int a, ...) requires AtLeast2<int> && true { // expected-note {{candidate function}}
    return 1;
  }

  template<typename T>
  constexpr int doo(int b) requires AtLeast2<int> { // expected-note {{candidate function}}
    return 2;
  }

  static_assert(goo<int>(1) == 1);
  static_assert(doo<int>(2) == 1); // expected-error {{call to 'doo' is ambiguous}}
}
