// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks

#include <stdarg.h>

int main() {
  void (^b) (int arg, const char * format, ...) __attribute__ ((__format__ (__printf__, 1, 3))) =   // expected-error {{format argument not a string type}}
    ^ __attribute__ ((__format__ (__printf__, 1, 3))) (int arg, const char * format, ...) {}; // expected-error {{format argument not a string type}}
 
  void (^z) (int arg, const char * format, ...) __attribute__ ((__format__ (__printf__, 2, 3))) = ^ __attribute__ ((__format__ (__printf__, 2, 3))) (int arg, const char * format, ...) {};

  z(1, "%s", 1); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  z(1, "%s", "HELLO"); // no-warning
}

void multi_attr(va_list ap, int *x, long *y) {
  // Handle block with multiple format attributes.
  void (^vprintf_scanf) (const char *, va_list, const char *, ...) __attribute__((__format__(__printf__, 1, 0))) __attribute__((__format__(__scanf__, 3, 4))) =
  ^ __attribute__((__format__(__printf__, 1, 0))) __attribute__((__format__(__scanf__, 3, 4))) (const char *str, va_list args, const char *fmt, ...) {};

  vprintf_scanf("%", ap, "%d"); // expected-warning {{incomplete format specifier}}, expected-warning {{more '%' conversions than data arguments}}
}
