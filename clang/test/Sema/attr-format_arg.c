// RUN: %clang_cc1 -fsyntax-only -verify %s

int printf(const char *, ...);

const char* f(const char *s) __attribute__((format_arg(1)));

const char *h(const char *msg1, const char *msg2)
    __attribute__((__format_arg__(1))) __attribute__((__format_arg__(2)));

void g(const char *s) {
  printf("%d", 123);
  printf("%d %d", 123); // expected-warning{{more '%' conversions than data arguments}}

  printf(f("%d"), 123);
  printf(f("%d %d"), 123); // expected-warning{{more '%' conversions than data arguments}}

  printf(h(
    "", // expected-warning {{format string is empty}}
    ""  // expected-warning {{format string is empty}}
  ), 123);
  printf(h(
    "%d",
    ""  // expected-warning {{format string is empty}}
  ), 123);
  printf(h(
    "", // expected-warning {{format string is empty}}
    "%d"
  ), 123);
  printf(h("%d", "%d"), 123);
}
