// RUN: %clang_cc1 %s -fsyntax-only -verify -fblocks
// rdar://10466373

typedef short SHORT;

void f0(void) {
  (void) ^{
    if (1)
      return (float)1.0;
    else if (2)
      return (double)2.0; // expected-error {{return type 'double' must match previous return type 'float' when block literal has}}
    else
      return (SHORT)3; // expected-error {{return type 'SHORT' (aka 'short') must match previous return type 'float' when}}
  };
}
