// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

using size_t = decltype(sizeof(0));
template<typename T> struct check;
template<size_t N> struct check<const char[N]> {};

constexpr bool startswith(const char *p, const char *q) {
  return !*q || (*p == *q && startswith(p + 1, q + 1));
}
constexpr bool contains(const char *p, const char *q) {
  return *p && (startswith(p, q) || contains(p + 1, q));
}

void foo() {
  check<decltype(__func__)>();
  static_assert(contains(__func__, "foo"), "");
}
