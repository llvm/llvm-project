// RUN: %clang_cc1 -triple x86_64-pc-linux -verify=expected,array-type -std=c2y -Wall -pedantic -ffreestanding %s
// RUN: %clang_cc1 -triple aarch64-pc-win32 -verify=expected,char-ptr-type -std=c2y -Wall -pedantic -ffreestanding %s

/* WG14 N3363: Clang 5
 * stdarg.h wording... v3
 *
 * Clarifies that va_start can only be used within a function body, and that
 * the macros and functions require an lvalue of type va_list instead of an
 * rvalue.
 *
 * Clang has diagnosed this correctly in all cases except va_copy since
 * Clang 5. va_copy still needs to be updated to issue a diagnostic about use
 * of an rvalue. However, our lack of a diagnostic is still conforming because
 * a diagnostic is not required (it's not a constraint violation).
 */

#include <stdarg.h>

// The va_list argument given to every macro defined in this subclause shall be
// an lvalue of this type or the result of array-to-pointer decay of such an
// lvalue.
void f(int a, ...) {
  va_list rvalue(); // array-type-error {{function cannot return array type 'va_list' (aka '__builtin_va_list')}}
  va_start(rvalue()); // / char-ptr-type-error {{non-const lvalue reference to type '__builtin_va_list' cannot bind to a temporary of type 'va_list' (aka 'char *')}}
  va_arg(rvalue(), int); // char-ptr-type-error {{expression is not assignable}}
  // FIXME: this should get two diagnostics about use of a non-lvalue.
  va_copy(rvalue(), rvalue()); // char-ptr-type-error {{non-const lvalue reference to type '__builtin_va_list' cannot bind to a temporary of type 'va_list' (aka 'char *')}}
  va_end(rvalue()); // char-ptr-type-error {{non-const lvalue reference to type '__builtin_va_list' cannot bind to a temporary of type 'va_list' (aka 'char *')}}
}

// The va_start macro may only be invoked in the compound-statement of the body
// of a variadic function.
void g(va_list ap, int [(va_start(ap), 1)], ...) { // expected-error {{'va_start' cannot be used outside a function}}
  va_end(ap);
}

va_list ap;
struct S {
  int array[(va_start(ap), 1)]; // expected-error {{'va_start' cannot be used outside a function}}
};
