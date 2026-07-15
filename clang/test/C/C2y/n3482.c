// RUN: %clang_cc1 -verify -std=c2y -Wall -pedantic -ffreestanding %s

/* WG14 N3482: Yes
 * Slay Some Earthly Demons XVII
 *
 * This paper makes it a constraint violation to call va_start in a non-
 * variadic function. This is something Clang has always diagnosed.
 */

#include <stdarg.h>

void func(int a) {
  va_list list;
  va_start(list, a); // expected-error {{'va_start' used in function with fixed args}}
}
