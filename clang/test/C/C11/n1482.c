// RUN: %clang_cc1 -verify -ffreestanding -std=c11 %s
// expected-no-diagnostics

/* WG14 N1482: Clang 4
 * Explicit initializers for atomics
 *
 * NB: We can only test the compile time behavior from the paper, not the
 * runtime behavior.
 */

#include <stdatomic.h>

#ifndef ATOMIC_BOOL_LOCK_FREE
#error "Missing ATOMIC_BOOL_LOCK_FREE"
#endif

#ifndef ATOMIC_CHAR_LOCK_FREE
#error "Missing ATOMIC_CHAR_LOCK_FREE"
#endif

#ifndef ATOMIC_CHAR16_T_LOCK_FREE
#error "Missing ATOMIC_CHAR16_T_LOCK_FREE"
#endif

#ifndef ATOMIC_CHAR32_T_LOCK_FREE
#error "Missing ATOMIC_CHAR32_T_LOCK_FREE"
#endif

#ifndef ATOMIC_WCHAR_T_LOCK_FREE
#error "Missing ATOMIC_WCHAR_T_LOCK_FREE"
#endif

#ifndef ATOMIC_SHORT_LOCK_FREE
#error "Missing ATOMIC_SHORT_LOCK_FREE"
#endif

#ifndef ATOMIC_INT_LOCK_FREE
#error "Missing ATOMIC_INT_LOCK_FREE"
#endif

#ifndef ATOMIC_LONG_LOCK_FREE
#error "Missing ATOMIC_LONG_LOCK_FREE"
#endif

#ifndef ATOMIC_LLONG_LOCK_FREE
#error "Missing ATOMIC_LLONG_LOCK_FREE"
#endif

#ifndef ATOMIC_POINTER_LOCK_FREE
#error "Missing ATOMIC_POINTER_LOCK_FREE"
#endif

#ifndef ATOMIC_VAR_INIT
#error "Missing ATOMIC_VAR_INIT"
#endif

#ifndef atomic_init
#error "Missing atomic_init"
#endif
