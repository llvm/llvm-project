// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s
@interface A {
  struct X {
    int x, y;
  } X;
}
@end

static const uint32_t Count = 16; // expected-error {{unknown type name 'uint32_t'}}

struct S0 {
  S0();
};

@interface C0
@end

@implementation C0 {
  S0 ivar0[Count];
}
@end
