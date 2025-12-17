// RUN: %clang_cc1 -verify -ffreestanding %s

/* WG14 ???: yes
 * restricted character set support via digraphs and <iso646.h>
 *
 * NB: I cannot find a definitive document number associated with the feature,
 * which was pulled from the editor's report in the C99 front matter. However,
 * based on discussion in the C99 rationale document, I believe this is
 * referring to features added by AMD1 to support ISO 646 and digraphs.
 */

// Validate that we provide iso646.h in freestanding mode.
#include <iso646.h>

// Validate that we define all the expected macros and their expected
// expansions (when suitable for a constant expression) as well.
#ifndef and
#error "missing and"
#else
_Static_assert((1 and 1) == (1 && 1), "");
#endif

#ifndef and_eq
#error "missing and_eq"
#endif

#ifndef bitand
#error "missing bitand"
#else
_Static_assert((1 bitand 3) == (1 & 3), "");
#endif

#ifndef bitor
#error "missing bitor"
#else
_Static_assert((1 bitor 2) == (1 | 2), "");
#endif

#ifndef compl
#error "missing compl"
#else
_Static_assert((compl 0) == (~0), "");
#endif

#ifndef not
#error "missing not"
#else
_Static_assert((not 12) == (!12), "");
#endif

#ifndef not_eq
#error "missing not_eq"
#else
_Static_assert((0 not_eq 12) == (0 != 12), "");
#endif

#ifndef or
#error "missing or"
#else
// This intentionally diagnoses use of '||' only, because the user likely did
// not confuse the operator when using 'or' instead.
_Static_assert((0 or 12) == (0 || 12), ""); // expected-warning {{use of logical '||' with constant operand}} \
                                               expected-note {{use '|' for a bitwise operation}}
#endif

#ifndef or_eq
#error "missing or_eq"
#endif

#ifndef xor
#error "missing xor"
#else
_Static_assert((1 xor 3) == (1 ^ 3), "");
#endif

#ifndef xor_eq
#error "missing xor_eq"
#endif

// Validate that digraphs behave the same as their expected counterparts. The
// definition should match the declaration in every way except spelling.
#define DI_NAME(f, b) f %:%: b
#define STD_NAME(f, b) f ## b
void DI_NAME(foo, bar)(int (*array)<: 0 :>);
void STD_NAME(foo, bar)(int (*array)[0]) {}

#define DI_STR(f) %:f
#define STD_STR(f) #f
_Static_assert(__builtin_strcmp(DI_STR(testing), STD_STR(testing)) == 0, "");

