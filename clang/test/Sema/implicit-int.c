// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic -Wno-strict-prototypes

foo(void) { // expected-warning {{type specifier missing, defaults to 'int'}}
  return 0;
}

y;  // expected-warning {{type specifier missing, defaults to 'int'}}

// rdar://6131634
void f((x));  // expected-warning {{type specifier missing, defaults to 'int'}}


// PR3702
#define PAD(ms10) { \
    register i;     \
}

#define ILPAD() PAD((NROW - tt.tt_row) * 10) /* 1 ms per char */

void
h19_insline(n)  // expected-warning {{parameter 'n' was not declared, defaulting to type 'int'}}
{
  ILPAD();  // expected-warning {{type specifier missing, defaults to 'int'}}
}

struct foo {
 __extension__ __attribute__((packed)) x : 4; // expected-warning {{type specifier missing, defaults to 'int'}}
};
