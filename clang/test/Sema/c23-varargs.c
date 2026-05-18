// RUN: %clang_cc1 -std=c23 -fsyntax-only -ffreestanding -verify=expected,both %s -triple i386-pc-unknown
// RUN: %clang_cc1 -std=c23 -fsyntax-only -ffreestanding -verify=expected,both %s -triple x86_64-apple-darwin9
// RUN: %clang_cc1 -std=c23 -fsyntax-only -ffreestanding -fms-compatibility -verify=expected,both %s -triple x86_64-pc-win32
// RUN: %clang_cc1 -std=c17 -fsyntax-only -ffreestanding -verify=both,pre-c23 %s

void foo(int x, int y, ...) {
  __builtin_va_list list;
  __builtin_c23_va_start();           // pre-c23-error {{use of unknown builtin '__builtin_c23_va_start'}} \
                                         expected-error{{too few arguments to function call, expected 1, have 0}}
  // Note, the unknown builtin diagnostic is only issued once per function,
  // which is why the rest of the lines do not get the same diagonstic.
  __builtin_c23_va_start(list);       // ok
  __builtin_c23_va_start(list, 0);    // expected-warning {{second argument to 'va_start' is not the last non-variadic parameter}}
  __builtin_c23_va_start(list, x);    // expected-warning {{second argument to 'va_start' is not the last non-variadic parameter}}
  __builtin_c23_va_start(list, y);    // ok
  __builtin_c23_va_start(list, 0, 1); // expected-error {{too many arguments to function call, expected at most 2, have 3}}
  __builtin_c23_va_start(list, y, y); // expected-error {{too many arguments to function call, expected at most 2, have 3}}
}

// Test the same thing as above, only with the macro from stdarg.h. This will
// not have the unknown builtin diagnostics, but will have different
// diagnostics between C23 and earlier modes.
#include <stdarg.h>
void bar(int x, int y, ...) {
  // FIXME: the "use of undeclared identifier 'va_start'" diagnostics is an odd
  // follow-on diagnostic that should be silenced.
  va_list list;
  va_start();           // pre-c23-error {{too few arguments provided to function-like macro invocation}} \
                           pre-c23-error {{use of undeclared identifier 'va_start'}} \
                           expected-error{{too few arguments to function call, expected 1, have 0}}
  va_start(list);       // pre-c23-error {{too few arguments provided to function-like macro invocation}} \
                           pre-c23-error {{use of undeclared identifier 'va_start'}}
  va_start(list, 0);    // both-warning {{second argument to 'va_start' is not the last non-variadic parameter}}
  va_start(list, x);    // both-warning {{second argument to 'va_start' is not the last non-variadic parameter}}
  va_start(list, y);    // ok
  va_start(list, 0, 1); // pre-c23-error {{too many arguments provided to function-like macro invocation}} \
                           pre-c23-error {{use of undeclared identifier 'va_start'}} \
                           expected-error {{too many arguments to function call, expected at most 2, have 3}}
  va_start(list, y, y); // pre-c23-error {{too many arguments provided to function-like macro invocation}} \
                           pre-c23-error {{use of undeclared identifier 'va_start'}} \
                           expected-error {{too many arguments to function call, expected at most 2, have 3}}	
  // pre-c23-note@__stdarg_va_arg.h:* 4 {{macro 'va_start' defined here}}
}
