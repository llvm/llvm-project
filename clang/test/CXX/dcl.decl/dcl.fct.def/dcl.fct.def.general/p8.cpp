// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

using size_t = decltype(sizeof(0));
template<typename T> struct check;
template<size_t N> struct check<const char[N]> {};

constexpr bool starts_with(const char *p, const char *q) {
  return !*q || (*p == *q && starts_with(p + 1, q + 1));
}
constexpr bool contains(const char *p, const char *q) {
  return *p && (starts_with(p, q) || contains(p + 1, q));
}

void foo() {
  check<decltype(__func__)>();
  static_assert(contains(__func__, "foo"), "");
}
