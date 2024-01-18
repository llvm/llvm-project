// RUN: %clang_cc1 -verify -std=c2x -ffreestanding %s

/* WG14 N2359: no
 * Remove conditional "WANT" macros from numbered clauses
 */

#include <limits.h>
#ifndef __STDC_VERSION_LIMITS_H__
#error "__STDC_VERSION_LIMITS_H__ not defined"
// expected-error@-1 {{"__STDC_VERSION_LIMITS_H__ not defined"}}
#endif

#include <stdarg.h>
#ifndef __STDC_VERSION_STDARG_H__
#error "__STDC_VERSION_STDARG_H__ not defined"
// expected-error@-1 {{"__STDC_VERSION_STDARG_H__ not defined"}}
#endif

#include <stdatomic.h>
#ifndef __STDC_VERSION_STDATOMIC_H__
#error "__STDC_VERSION_STDATOMIC_H__ not defined"
// expected-error@-1 {{"__STDC_VERSION_STDATOMIC_H__ not defined"}}
#endif

#include <stddef.h>
#ifndef __STDC_VERSION_STDDEF_H__
#error "__STDC_VERSION_STDDEF_H__ not defined"
// expected-error@-1 {{"__STDC_VERSION_STDDEF_H__ not defined"}}
#endif

#include <stdint.h>
#ifndef __STDC_VERSION_STDINT_H__
#error "__STDC_VERSION_STDINT_H__ not defined"
// expected-error@-1 {{"__STDC_VERSION_STDINT_H__ not defined"}}
#endif

#include <stdckdint.h>
#ifndef __STDC_VERSION_STDCKDINT_H__
#error "__STDC_VERSION_STDCKDINT_H__ not defined"
#endif
