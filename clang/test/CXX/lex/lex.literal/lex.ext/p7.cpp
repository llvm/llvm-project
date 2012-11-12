// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s

using size_t = decltype(sizeof(int));
namespace std {
  struct string {};
}

template<typename T, typename U> struct same_type;
template<typename T> struct same_type<T, T> {};

namespace std_example {

long double operator "" _w(long double);
std::string operator "" _w(const char16_t*, size_t);
unsigned operator "" _w(const char*);
int main() {
  auto v1 = 1.2_w;    // calls operator "" _w(1.2L)
  auto v2 = u"one"_w; // calls operator "" _w(u"one", 3)
  auto v3 = 12_w;     // calls operator "" _w("12")
  "two"_w;            // expected-error {{no matching literal operator}}

  same_type<decltype(v1), long double> test1;
  same_type<decltype(v2), std::string> test2;
  same_type<decltype(v3), unsigned> test3;
}

}
