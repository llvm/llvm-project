// RUN: %clang_cc1 -fsyntax-only -verify -fdiagnostics-parseable-fixits %s -Wgnu-zero-variadic-macro-arguments -std=c23

void foo(const char* fmt, ...);
// expected-warning@+1 {{token pasting of ',' and __VA_ARGS__ is a GNU extension. Consider using __VA_OPT__(,) instead}}
#define FOO(format, ...) foo(format, ##__VA_ARGS__)

void bar(void) {
  FOO("", 0);
}
