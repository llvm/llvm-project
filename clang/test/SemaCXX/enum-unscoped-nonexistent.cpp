// RUN: %clang_cc1 -std=c++11 -verify %s

struct Base {
  static const int a = 1;
};
template<typename T> struct S : Base {
  enum E : int;
  constexpr int f() const;
  constexpr int g() const;
  void h();
};
template<> enum S<char>::E : int {}; // expected-note {{enum 'S<char>::E' was explicitly specialized here}}
template<> enum S<short>::E : int { b = 2 };
template<> enum S<int>::E : int { a = 4 };
template<typename T> enum S<T>::E : int { b = 8 };

// The unqualified-id here names a member of the non-dependent base class Base
// and not the injected enumerator name 'a' from the specialization.
template<typename T> constexpr int S<T>::f() const { return a; }
static_assert(S<char>().f() == 1, "");
static_assert(S<int>().f() == 1, "");

// The unqualified-id here names a member of the current instantiation, which
// bizarrely might not exist in some instantiations.
template<typename T> constexpr int S<T>::g() const { return b; } // expected-error {{enumerator 'b' does not exist in instantiation of 'S<char>'}}
static_assert(S<char>().g() == 1, ""); // expected-note {{here}} expected-error {{not an integral constant expression}}
static_assert(S<short>().g() == 2, "");
static_assert(S<long>().g() == 8, "");

// 'b' is type-dependent, so these assertions should not fire before 'h' is
// instantiated.
template<typename T> void S<T>::h() {
  char c[S<T>::b];
  static_assert(b != 8, "");
  static_assert(sizeof(c) != 8, "");
}
void f() {
  S<short>().h(); // ok, b == 2
}
