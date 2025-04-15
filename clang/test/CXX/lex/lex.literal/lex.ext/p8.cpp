// RUN: %clang_cc1 -std=c++11 -verify %s

using size_t = decltype(sizeof(int));
constexpr const char *operator ""_id(const char *p, size_t) { return p; }
constexpr const char *s = "foo"_id "bar" "baz"_id "quux";

constexpr bool streq(const char *p, const char *q) {
  return *p == *q && (!*p || streq(p+1, q+1));
}
static_assert(streq(s, "foobarbazquux"), "");

constexpr const char *operator ""_trim(const char *p, size_t n) {
  return *p == ' ' ? operator ""_trim(p + 1, n - 1) : p;
}
constexpr const char *t = "   " " "_trim "  foo";
static_assert(streq(t, "foo"), "");

const char *u = "foo" "bar"_id "baz" "quux"_di "corge"; // expected-error {{differing user-defined suffixes ('_id' and '_di') in string literal concatenation}}
