// RUN: %clang_cc1 -std=c++11 %s -verify

using size_t = decltype(sizeof(int));
template<char...> struct S {};

template<char...> void operator "" _a();
template<char... C> S<C...> operator "" _a();

template<typename T> struct U {
  friend int operator "" _a(const char *, size_t);
  // FIXME: It's not entirely clear whether this is intended to be legal.
  friend U operator "" _a(const T *, size_t); // expected-error {{parameter}}
};
template<char...> struct V {
  friend void operator "" _b(); // expected-error {{parameters}}
};

template<char... C, int N = 0> void operator "" _b(); // expected-error {{template}}
template<char... C> void operator "" _b(int N = 0); // expected-error {{template}}
template<char, char...> void operator "" _b(); // expected-error {{template}}
template<typename T> T operator "" _b(const char *); // expected-error {{template}}
template<typename T> int operator "" _b(const T *, size_t); // expected-error {{template}}
