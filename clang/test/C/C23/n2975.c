// RUN: %clang_cc1 -verify -ffreestanding -Wpre-c2x-compat -std=c2x %s

/* WG14 N2975: partial
 * Relax requirements for va_start
 */

#include <stdarg.h>

#define DERP this is an error

void func(...) { // expected-warning {{'...' as the only parameter of a function is incompatible with C standards before C23}}
  // Show that va_start doesn't require the second argument in C23 mode.
  va_list list;
  va_start(list); // expected-warning {{passing no argument for the '...' parameter of a variadic macro is incompatible with C standards before C23}} expected-note@* {{macro 'va_start' defined here}}
  va_end(list);

  // Show that va_start doesn't expand or evaluate the second argument.
  va_start(list, DERP);
  va_end(list);

  // FIXME: it would be kinder to diagnose this instead of silently accepting it.
  va_start(list, 1, 2);
  va_end(list);

  // We didn't change the behavior of __builtin_va_start (and neither did GCC).
  __builtin_va_start(list); // expected-error {{too few arguments to function call, expected 2, have 1}}

  // Verify that the return type of a call to va_start is 'void'.
  _Static_assert(__builtin_types_compatible_p(__typeof__(va_start(list)), void), ""); // expected-warning {{passing no argument for the '...' parameter of a variadic macro is incompatible with C standards before C23}} expected-note@* {{macro 'va_start' defined here}}
  _Static_assert(__builtin_types_compatible_p(__typeof__(__builtin_va_start(list, 0)), void), "");
}

// Show that function pointer types also don't need an argument before the
// ellipsis.
typedef void (*fp)(...); // expected-warning {{'...' as the only parameter of a function is incompatible with C standards before C23}}

// Passing something other than the argument before the ... is still not valid.
void diag(int a, int b, ...) {
  va_list list;
  // FIXME: the call to va_start should also diagnose the same way as the call
  // to __builtin_va_start. However, because va_start is not allowed to expand
  // or evaluate the second argument, we can't pass it along to
  // __builtin_va_start to get that diagnostic. So in C17 and earlier, we will
  // diagnose this use through the macro, but in C23 and later we've lost the
  // diagnostic entirely. GCC has the same issue currently.
  va_start(list, a);
  // However, the builtin itself is under no such constraints regarding
  // expanding or evaluating the second argument, so it can still diagnose.
  __builtin_va_start(list, a); // expected-warning {{second argument to 'va_start' is not the last named parameter}}
  va_end(list);
}

void foo(int a...); // expected-error {{C requires a comma prior to the ellipsis in a variadic function type}}

void use(void) {
  // Demonstrate that we can actually call the variadic function when it has no
  // formal parameters.
  func(1, '2', 3.0, "4");
  func();

  // And that assignment still works as expected.
  fp local = func;

  // ...including conversion errors.
  fp other_local = diag; // expected-error {{incompatible function pointer types initializing 'fp' (aka 'void (*)(...)') with an expression of type 'void (int, int, ...)'}}
}
