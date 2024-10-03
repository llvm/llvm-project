// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

constexpr int operator ""_a(const char *c) {
  return c[0];
}

static_assert(operator ""_a("foo") == 'f', "");

void puts(const char *);
static inline void operator ""_puts(const char *c) {
  puts(c);
}
void f() {
  operator ""_puts("foo");
  operator ""_puts("bar");
}
