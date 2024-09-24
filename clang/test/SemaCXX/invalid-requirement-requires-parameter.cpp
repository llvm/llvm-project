// RUN: %clang -fsyntax-only -std=c++2a -Xclang -verify %s

namespace GH109538 {
static_assert(requires(void *t) { t; });
static_assert(requires(void) { 42; });
static_assert(requires(void t) { // expected-error {{argument may not have 'void' type}}
  t;
});
static_assert(requires(void t, int a) {  // expected-error {{'void' must be the first and only parameter if specified}}
  t;
});
static_assert(requires(const void) { // expected-error {{'void' as parameter must not have type qualifiers}}
  42;
});
} // namespace GH109538
