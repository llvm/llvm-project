// RUN: %clang_cc1 %s -verify -fsyntax-only -Wc++11-compat -std=c++98

#define constexpr const
constexpr int x = 0;
#undef constexpr

namespace lib {
  struct nullptr_t;
  typedef nullptr_t nullptr; // expected-warning {{'nullptr' is a keyword in C++11}}
}

#define CONCAT(X,Y) CONCAT2(X,Y)
#define CONCAT2(X,Y) X ## Y
int CONCAT(constexpr,ession);

#define ID(X) X
extern int ID(decltype); // expected-warning {{'decltype' is a keyword in C++11}}

extern int CONCAT(align,of); // expected-warning {{'alignof' is a keyword in C++11}}

#define static_assert(b, s) int CONCAT(check, __LINE__)[(b) ? 1 : 0];
static_assert(1 > 0, "hello"); // ok

#define IF_CXX11(CXX11, CXX03) CXX03
typedef IF_CXX11(char16_t, wchar_t) my_wide_char_t; // ok

int alignas; // expected-warning {{'alignas' is a keyword in C++11}}
int alignof; // already diagnosed in this TU
int char16_t; // expected-warning {{'char16_t' is a keyword in C++11}}
int char32_t; // expected-warning {{'char32_t' is a keyword in C++11}}
int constexpr; // expected-warning {{'constexpr' is a keyword in C++11}}
int decltype; // already diagnosed in this TU
int noexcept; // expected-warning {{'noexcept' is a keyword in C++11}}
int nullptr; // already diagnosed in this TU
int static_assert; // expected-warning {{'static_assert' is a keyword in C++11}}
int thread_local; // expected-warning {{'thread_local' is a keyword in C++11}}
