// RUN: %clang_cc1 -ffreestanding -verify=expected,c2x -std=c2x -Wpre-c2x-compat %s
// RUN: %clang_cc1 -ffreestanding -verify=expected,c17 -std=c17 %s

/* WG14 N2934: yes
 * Revise spelling of keywords v7
 */

thread_local struct S { // c2x-warning {{'thread_local' is incompatible with C standards before C23}} \
                           c2x-error 0+ {{thread-local storage is not supported for the current target}} \
                           c17-error {{unknown type name 'thread_local'}}
  bool b; // c2x-warning {{'bool' is incompatible with C standards before C23}} \
             c17-error {{unknown type name 'bool'}}
} s;

static_assert(alignof(int) != 0, ""); // c2x-warning {{'static_assert' is incompatible with C standards before C23}} \
                                         c2x-warning {{'alignof' is incompatible with C standards before C23}} \
                                         c17-error 2 {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                         c17-error {{expected ')'}} \
                                         c17-note {{to match this '('}}

#include <stdalign.h>

// C17 and earlier must have __alignas_is_defined and __alignof_is_defined,
// but C2x and later must not.
#if __STDC_VERSION__ <= 201710L
  #if __alignas_is_defined != 1
    #error "alignas should be defined"
  #endif
  #if __alignof_is_defined != 1
    #error "alignof should be defined"
  #endif
#else
  #ifdef __alignas_is_defined
    #error "alignas should not be defined"
  #endif
  #ifdef __alignof_is_defined
    #error "alignof should not be defined"
  #endif
#endif

#include <stdbool.h>

// C17 and earlier must have bool defined as a macro, but C2x and later should
// not (at least in Clang's implementation; it's permissible for bool to be a
// macro in general, as it could expand to _Bool).
#if __STDC_VERSION__ <= 201710L
  #ifndef bool
    #error "bool should be defined"
  #endif
#else
  #ifdef bool
    #error "bool should not be defined"
  #endif
#endif

// Ensure we correctly parse the alignas keyword in a specifier-qualifier-list.
// This is different than in C++ where alignas is an actual attribute rather
// than a specifier.
struct GH81472 {
  char alignas(8) a1;   // c2x-warning {{'alignas' is incompatible with C standards before C23}}
  alignas(8) char a2;   // c2x-warning {{'alignas' is incompatible with C standards before C23}}
  char _Alignas(8) a3;
  _Alignas(8) char a4;
  char a5 alignas(8);   // expected-error {{expected ';' at end of declaration list}}
  char a6 _Alignas(8);  // expected-error {{expected ';' at end of declaration list}}
};

// Ensure we reject alignas as an attribute specifier. This code is accepted in
// C++ mode but should be rejected in C.
// FIXME: this diagnostic could be improved
struct alignas(8) Reject1 { // expected-error {{declaration of anonymous struct must be a definition}} \
                               expected-warning {{declaration does not declare anything}}
  int a;
};

struct _Alignas(8) Reject2 { // expected-error {{declaration of anonymous struct must be a definition}} \
                                expected-warning {{declaration does not declare anything}}
  int a;
};
