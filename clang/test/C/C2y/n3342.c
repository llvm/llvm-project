// RUN: %clang_cc1 -verify=expected,both -std=c2y -Wall -pedantic %s
// RUN: %clang_cc1 -verify=clang,both -Wall -pedantic %s

/* WG14 N3342: Yes
 * Slay Some Earthly Demons IV
 *
 * Qualified function types are now implementation-defined instead of
 * undefined. Clang strips the qualifiers.
 */

typedef int f(void);

const f one;      /* expected-warning {{'const' qualifier on function type 'f' (aka 'int (void)') has no effect}}
                     clang-warning {{'const' qualifier on function type 'f' (aka 'int (void)') has no effect and is a Clang extension}}
                   */
volatile f two;   /* expected-warning {{'volatile' qualifier on function type 'f' (aka 'int (void)') has no effect}}
                     clang-warning {{'volatile' qualifier on function type 'f' (aka 'int (void)') has no effect and is a Clang extension}}
                   */

const volatile f three; /* expected-warning {{'const' qualifier on function type 'f' (aka 'int (void)') has no effect}}
                           clang-warning {{'const' qualifier on function type 'f' (aka 'int (void)') has no effect and is a Clang extension}}
                           expected-warning {{'volatile' qualifier on function type 'f' (aka 'int (void)') has no effect}}
                           clang-warning {{'volatile' qualifier on function type 'f' (aka 'int (void)') has no effect and is a Clang extension}}
                         */

#if __STDC_VERSION__ >= 201112L
// Atomic types have an explicit constraint making it ill-formed.
_Atomic f four;   // both-error {{_Atomic cannot be applied to function type 'f' (aka 'int (void)')}}
#endif

// There's no point to testing 'restrict' because that requires a pointer type.
