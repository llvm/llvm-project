// RUN: %clang_cc1 %s -verify -fsyntax-only -fno-dollars-in-identifiers

#define f(x) #x
#define g(x) f(x)
#define foo bar

constexpr auto expanded = g($foo);
static_assert(expanded[0] == '$'
           && expanded[1] == 'f'
           && expanded[2] == 'o'
           && expanded[3] == 'o');

#define $test 1
int a = $test;

int $b = 2; // expected-error {{identifier '$b' is not supported with the current configuration}}

#define name $name
int name = 2;
// expected-error@-1 {{identifier '$name' is not supported with the current configuration}}

int $zoinks() { // expected-error {{identifier '$zoinks' is not supported with the current configuration}}
  return $y + name;
  // expected-error@-1 {{identifier '$y' is not supported with the current configuration}} \
  // expected-error@-1 {{use of undeclared identifier '$y'}} \
  // expected-error@-1 {{identifier '$name' is not supported with the current configuration}}
}
