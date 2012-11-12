// RUN: %clang_cc1 -analyze -analyzer-checker=core -analyzer-store=region -fblocks -verify %s

typedef void (^bptr)(void);

bptr bf(int j) {
  __block int i;
  const bptr &qq = ^{ i=0; }; // expected-note {{binding reference variable 'qq' here}}
  return qq; // expected-error {{returning block that lives on the local stack}}
}
