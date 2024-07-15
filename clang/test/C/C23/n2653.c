// RUN: %clang_cc1 -verify=pre-c23 -ffreestanding -std=c17 %s
// RUD: %clang_cc1 -verify -ffreestanding -std=c23 %s

/* WG14 N2653: Clang 19
 * char8_t: A type for UTF-8 characters and strings
 */

// expected-no-diagnostics

#include <stdatomic.h>

typedef unsigned char char8_t;  // in <uchar.h>, which Clang does not provide.

#if __STDC_VERSION__ >= 202311L
  #define LITERAL_TYPE char8_t
  #define LITERAL_UNDERLYING_TYPE unsigned char

  // Ensure that char8_t has the same lock-free capabilities as unsigned char.
  #if defined(ATOMIC_CHAR8_T_LOCK_FREE) != defined(ATOMIC_CHAR_LOCK_FREE) || \
      ATOMIC_CHAR8_T_LOCK_FREE != ATOMIC_CHAR_LOCK_FREE
    #error "invalid char8_t atomic lock free status"
  #endif

#else
  #define LITERAL_TYPE char
  #define LITERAL_UNDERLYING_TYPE char

  // Ensure we don't define the lock-free status in earlier modes.
  #if defined(ATOMIC_CHAR8_T_LOCK_FREE)
    #error "ATOMIC_CHAR8_T_LOCK_FREE should not be defined"
  #endif
#endif

// Ensure we get the type of the literal correct.
_Static_assert(_Generic(u8""[0], LITERAL_TYPE : 1, default : 0), "");
_Static_assert(_Generic(u8""[0], LITERAL_UNDERLYING_TYPE : 1, default : 0), "");

// Ensure we have a datatype for atomic operations.
atomic_char8_t val; // pre-c23-error {{unknown type name 'atomic_char8_t'}}
