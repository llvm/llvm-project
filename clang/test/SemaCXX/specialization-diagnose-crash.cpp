// RUN: %clang_cc1 -fsyntax-only %s --std=c++17 -verify
// This is a reduction of GH57370 and GH58028, originally appearing
// in libstdc++'s variant code.

struct V1 {};
struct V2 : V1 {
  int &a;
};

template <class T> using void_t = void;

template <class T> struct X { T x; };

template <class T1, class T2, class = void> struct Variant {
  Variant() = delete; // expected-note {{deleted here}}
};

template <class T1, class T2>
struct Variant<T1, T2, void_t<decltype(X<T2>{T1()})>> {};

void f() {
  Variant<V1, V1>();
  Variant<V1, V2>(); // expected-error {{call to deleted constructor}}
}
