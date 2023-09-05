// RUN: %clang_cc1 -ffreestanding -verify=c2x -std=c2x -Wpre-c2x-compat %s
// RUN: %clang_cc1 -ffreestanding -verify=c17 -std=c17 %s

/* WG14 N2934: yes
 * Revise spelling of keywords v7
 */

thread_local struct alignas(int) S { // c2x-warning {{'alignas' is incompatible with C standards before C23}} \
                                        c2x-warning {{'thread_local' is incompatible with C standards before C23}} \
                                        c2x-error 0+ {{thread-local storage is not supported for the current target}} \
                                        c17-error {{unknown type name 'thread_local'}} \
                                        c17-error {{expected identifier or '('}} \
                                        c17-error {{expected ')'}} \
                                        c17-note {{to match this '('}}
  bool b; // c2x-warning {{'bool' is incompatible with C standards before C23}}
} s; // c17-error {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}}

static_assert(alignof(struct S) == alignof(int), ""); // c2x-warning {{'static_assert' is incompatible with C standards before C23}} \
                                                         c2x-warning 2 {{'alignof' is incompatible with C standards before C23}} \
                                                         c17-error 2 {{type specifier missing, defaults to 'int'; ISO C99 and later do not support implicit int}} \
                                                         c17-error {{expected ')'}} \
                                                         c17-warning {{declaration of 'struct S' will not be visible outside of this function}} \
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
