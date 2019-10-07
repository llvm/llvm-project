// RUN: %clang_cc1 -triple x86_64-apple-darwin9.0 -fsyntax-only -verify -Wno-format -Wformat-type-confusion %s

__attribute__((format(__printf__, 1, 2)))
int printf(const char *msg, ...);

#define FMT "%hd %hu %d %u %hhd %hhu %c"

int main() {
  _Bool b = 0;
  printf(FMT,
         b, // expected-warning {{format specifies type 'short' but the argument has type '_Bool'}}
         b, // expected-warning {{format specifies type 'unsigned short' but the argument has type '_Bool'}}
         b, b, b, b, b);

  unsigned char uc = 0;
  printf(FMT,
         uc, // expected-warning {{format specifies type 'short' but the argument has type 'unsigned char'}}
         uc, // expected-warning {{format specifies type 'unsigned short' but the argument has type 'unsigned char'}}
         uc, uc, uc, uc, uc);

  signed char sc = 0;
  printf(FMT,
         sc, // expected-warning {{format specifies type 'short' but the argument has type 'signed char'}}
         sc, // expected-warning {{format specifies type 'unsigned short' but the argument has type 'signed char'}}
         sc, sc, sc, sc, sc);
}
