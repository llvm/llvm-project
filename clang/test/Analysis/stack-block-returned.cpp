// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store=region -fblocks -verify %s

typedef void (^bptr)(void);

bptr bf(int j) {
  __block int i;
  const bptr &qq = ^{ i=0; }; // expected-note {{binding reference variable 'qq' here}}
  return qq; // expected-error {{returning block that lives on the local stack}}
}
