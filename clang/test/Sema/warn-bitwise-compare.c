// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-bitwise-compare %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wall -Wno-unused %s

#define mydefine 2
#define mydefine2 -2

enum {
  ZERO,
  ONE,
};

void f(int x) {
  if ((8 & x) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x & 8) == 4) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((-8 & x) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x & -8) == 4) {}  // expected-warning {{bitwise comparison always evaluates to false}}

  if ((x & 8) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((2 & x) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((x & -8) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((-2 & x) != 3) {}  // expected-warning {{bitwise comparison always evaluates to true}}

  if ((x | 4) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x | 3) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((5 | x) != 3) {}  // expected-warning {{bitwise comparison always evaluates to true}}

  if ((x | -4) == 3) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x | -3) != 4) {}  // expected-warning {{bitwise comparison always evaluates to true}}
  if ((-5 | x) != 3) {}  // expected-warning {{bitwise comparison always evaluates to true}}


  if ((x & 0x15) == 0x13) {} // expected-warning {{bitwise comparison always evaluates to false}}
  if ((x & 0xFFEB) == 0x13) {} // expected-warning {{bitwise comparison always evaluates to false}}

  if ((0x23 | x) == 0x155){} // expected-warning {{bitwise comparison always evaluates to false}}
  if ((0xFFDD | x) == 0x155){} // expected-warning {{bitwise comparison always evaluates to false}}

  if (!!((8 & x) == 3)) {}  // expected-warning {{bitwise comparison always evaluates to false}}
  if (!!((-8 & x) == 3)) {}  // expected-warning {{bitwise comparison always evaluates to false}}

  int y = ((8 & x) == 3) ? 1 : 2;  // expected-warning {{bitwise comparison always evaluates to false}}
      y = ((-8 & x) == 3) ? 1 : 2;  // expected-warning {{bitwise comparison always evaluates to false}}
      y = ((3 | x) != 5) ? 1 : 2;  // expected-warning {{bitwise comparison always evaluates to true}}
      y = ((-3 | x) != 5) ? 1 : 2;  // expected-warning {{bitwise comparison always evaluates to true}}

  if ((x & 8) == 8) {}
  if ((x & 8) != 8) {}
  if ((x | 4) == 4) {}
  if ((x | 4) != 4) {}

  if ((-2 & x) != 4) {}
  if ((x & -8) == -8) {}
  if ((x & -8) != -8) {}
  if ((x | -4) == -4) {}
  if ((x | -4) != -4) {}


  if ((x & 9) == 8) {}
  if ((x & 9) != 8) {}
  if ((x | 4) == 5) {}
  if ((x | 4) != 5) {}

  if ((x & -9) == -10) {}
  if ((x & -9) != -10) {}
  if ((x | -4) == -3) {}
  if ((x | -4) != -3) {}

  if ((x^0) == 0) {}

  if ((x & mydefine) == 8) {}
  if ((x | mydefine) == 4) {}

  if ((x & mydefine2) == 8) {}
  if ((x | mydefine2) == 4) {}

}

void g(int x) {
  if (x | 5) {}  // expected-warning {{bitwise or with non-zero value always evaluates to true}}
  if (5 | x) {}  // expected-warning {{bitwise or with non-zero value always evaluates to true}}
  if (!((x | 5))) {}  // expected-warning {{bitwise or with non-zero value always evaluates to true}}

  if (x | -1) {}  // expected-warning {{bitwise or with non-zero value always evaluates to true}}
  if (x | ONE) {}  // expected-warning {{bitwise or with non-zero value always evaluates to true}}

  if (x | ZERO) {}
}
