// RUN: %clang_cc1 -fsyntax-only -verify %s

int printf(const char *, ...);

const char* f(const char *s) __attribute__((format_arg(1)));

void g(const char *s) {
  printf("%d", 123);
  printf("%d %d", 123); // expected-warning{{more '%' conversions than data arguments}}

  printf(f("%d"), 123);
  printf(f("%d %d"), 123); // expected-warning{{more '%' conversions than data arguments}}
}
