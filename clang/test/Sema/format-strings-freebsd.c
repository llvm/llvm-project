// RUN: %clang_cc1 -fsyntax-only -verify -triple i386-unknown-freebsd %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-freebsd %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-scei-ps4 %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-sie-ps5 %s

// Test FreeBSD kernel printf extensions.
int freebsd_kernel_printf(const char *, ...) __attribute__((__format__(__freebsd_kprintf__, 1, 2)));

void check_freebsd_kernel_extensions(int i, long l, char *s, short h)
{
  // %b expects an int and a char *
  freebsd_kernel_printf("reg=%b\n", i, "\10\2BITTWO\1BITONE\n"); // no-warning
  freebsd_kernel_printf("reg=%b\n", l, "\10\2BITTWO\1BITONE\n"); // expected-warning{{format specifies type 'int' but the argument has type 'long'}}
  freebsd_kernel_printf("reg=%b\n", i, l); // expected-warning{{format specifies type 'char *' but the argument has type 'long'}}
  freebsd_kernel_printf("reg=%b\n", i); // expected-warning{{more '%' conversions than data arguments}}
  freebsd_kernel_printf("reg=%b\n", i, "\10\2BITTWO\1BITONE\n", l); // expected-warning{{data argument not used by format string}}

  // %D expects an unsigned char * and a char *
  freebsd_kernel_printf("%6D", s, ":"); // no-warning
  freebsd_kernel_printf("%6D", i, ":"); // expected-warning{{format specifies type 'void *' but the argument has type 'int'}}
  freebsd_kernel_printf("%6D", s, i); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  freebsd_kernel_printf("%6D", s); // expected-warning{{more '%' conversions than data arguments}}
  freebsd_kernel_printf("%6D", s, ":", i); // expected-warning{{data argument not used by format string}}

  freebsd_kernel_printf("%*D", 42, s, ":"); // no-warning
  freebsd_kernel_printf("%*D", 42, i, ":"); // expected-warning{{format specifies type 'void *' but the argument has type 'int'}}
  freebsd_kernel_printf("%*D", 42, s, i); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  freebsd_kernel_printf("%*D", 42, s); // expected-warning{{more '%' conversions than data arguments}}
  freebsd_kernel_printf("%*D", 42, s, ":", i); // expected-warning{{data argument not used by format string}}

  // %r expects an int
  freebsd_kernel_printf("%r", i); // no-warning
  freebsd_kernel_printf("%r", l); // expected-warning{{format specifies type 'int' but the argument has type 'long'}}
  freebsd_kernel_printf("%lr", i); // expected-warning{{format specifies type 'long' but the argument has type 'int'}}
  freebsd_kernel_printf("%lr", l); // no-warning

  // h modifier expects a short
  freebsd_kernel_printf("%hr", i); // no-warning
  freebsd_kernel_printf("%hr", h); // no-warning
  freebsd_kernel_printf("%hy", i); // no-warning
  freebsd_kernel_printf("%hy", h); // no-warning

  // %y expects an int
  freebsd_kernel_printf("%y", i); // no-warning
  freebsd_kernel_printf("%y", l); // expected-warning{{format specifies type 'int' but the argument has type 'long'}}
  freebsd_kernel_printf("%ly", i); // expected-warning{{format specifies type 'long' but the argument has type 'int'}}
  freebsd_kernel_printf("%ly", l); // no-warning
}
