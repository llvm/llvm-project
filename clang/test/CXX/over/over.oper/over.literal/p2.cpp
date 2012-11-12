// RUN: %clang_cc1 -std=c++11 %s -verify

void operator "" _a(const char *);

namespace N {
  using ::operator "" _a;

  void operator "" _b(const char *);
}

using N::operator "" _b;

class C {
  void operator "" _c(const char *); // expected-error {{must be in a namespace or global scope}}

  static void operator "" _c(unsigned long long); // expected-error {{must be in a namespace or global scope}}

  friend void operator "" _d(const char *);
};

int operator "" _e; // expected-error {{cannot be the name of a variable}}

void f() {
  int operator "" _f; // expected-error {{cannot be the name of a variable}}
}

extern "C++" {
  void operator "" _g(const char *);
}

template<char...> void operator "" _h() {}

template<> void operator "" _h<'a', 'b', 'c'>() {}

template void operator "" _h<'a', 'b', 'c', 'd'>();
