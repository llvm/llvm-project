// RUN: %clang_cc1 %s -fsyntax-only -fno-dollars-in-identifiers -fdollars-in-macros -verify=expected
// RUN: %clang_cc1 %s -fsyntax-only -fno-dollars-in-identifiers -fno-dollars-in-macros -verify=old
// RUN: %clang_cc1 %s -fsyntax-only -fdollars-in-identifiers -fno-dollars-in-macros
// RUN: %clang_cc1 %s -fsyntax-only -fdollars-in-identifiers -fdollars-in-macros

#define f(x) #x
#define g(x) f(x)
#define foo bar

constexpr auto expanded = g($foo);
static_assert(expanded[0] == '$'
           && expanded[1] == 'f'
           && expanded[2] == 'o'
           && expanded[3] == 'o');
// old-error@-3 {{static assertion failed due to requirement 'expanded[1] == 'f''}} \
// old-note@-3 {{expression evaluates to ''b' (0x62, 98) == 'f' (0x66, 102)'}}

#define $test 1 // old-error {{macro name must be an identifier}}
int a = $test; // old-error {{expected expression}}

int $b = 2; 
// expected-error@-1 {{identifier '$b' is not supported with the current configuration}} \
// old-error@-1 {{expected unqualified-id}}

#define name $name
int name = 2;
// expected-error@-1 {{identifier '$name' is not supported with the current configuration}} \
// old-error@-1 {{expected unqualified-id}}

int $zoinks() { 
// expected-error@-1 {{identifier '$zoinks' is not supported with the current configuration}} \
// old-error@-1 {{expected unqualified-id}}
  return $b + name;
  // expected-error@-1 {{identifier '$b' is not supported with the current configuration}} \
  // expected-error@-1 {{identifier '$name' is not supported with the current configuration}}
}
