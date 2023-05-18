/* RUN: %clang_cc1 -std=c99 -ffreestanding -triple x86_64-unknown-linux -fsyntax-only -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c99 -ffreestanding -triple x86_64-unknown-win32 -fms-compatibility -fsyntax-only -verify -pedantic -Wno-c11-extensions %s
   RUN: %clang_cc1 -std=c11 -ffreestanding -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c17 -ffreestanding -fsyntax-only -verify -pedantic %s
   RUN: %clang_cc1 -std=c2x -ffreestanding -fsyntax-only -verify -pedantic %s
 */

/* WG14 DR209: partial
 * Problem implementing INTN_C macros
 */
#include <stdint.h>

#if INT8_C(0) != 0
#error "uh oh"
#elif INT16_C(0) != 0
#error "uh oh"
#elif INT32_C(0) != 0
#error "uh oh"
#elif INT64_C(0) != 0LL
#error "uh oh"
#elif UINT8_C(0) != 0U
#error "uh oh"
#elif UINT16_C(0) != 0U
#error "uh oh"
#elif UINT32_C(0) != 0U
#error "uh oh"
#elif UINT64_C(0) != 0ULL
#error "uh oh"
#endif

void dr209(void) {
  (void)_Generic(INT8_C(0), __typeof__(+(int_least8_t){0}) : 1);
  (void)_Generic(INT16_C(0), __typeof__(+(int_least16_t){0}) : 1);
  (void)_Generic(INT32_C(0), __typeof__(+(int_least32_t){0}) : 1);
  (void)_Generic(INT64_C(0), __typeof__(+(int_least64_t){0}) : 1);
  // FIXME: This is not the expected behavior; the type of the expanded value
  // in both of these cases should be 'int',
  //
  // C99 7.18.4p3: The type of the expression shall have the same type as would
  // an expression of the corresponding type converted according to the integer
  // promotions.
  //
  // C99 7.18.4.1p1: The macro UINTN_C(value) shall expand to an integer
  // constant expression corresponding to the type uint_leastN_t.
  //
  // C99 7.18.1.2p2: The typedef name uint_leastN_t designates an unsigned
  // integer type with a width of at least N, ...
  //
  // So the value's type is the same underlying type as uint_leastN_t, which is
  // unsigned char for uint_least8_t, and unsigned short for uint_least16_t,
  // but then the value undergoes integer promotions which would convert both
  // of those types to int.
  //
  (void)_Generic(UINT8_C(0), __typeof__(+(uint_least8_t){0}) : 1);
  (void)_Generic(UINT16_C(0), __typeof__(+(uint_least16_t){0}) : 1);
  // expected-error@-2 {{controlling expression type 'unsigned int' not compatible with any generic association type}}
  // expected-error@-2 {{controlling expression type 'unsigned int' not compatible with any generic association type}}
  (void)_Generic(UINT32_C(0), __typeof__(+(uint_least32_t){0}) : 1);
  (void)_Generic(UINT64_C(0), __typeof__(+(uint_least64_t){0}) : 1);
}

