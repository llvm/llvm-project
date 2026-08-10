// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface A {
  id xxx;
}
-(int)bar;
@end
@implementation A
-(int)bar {
  return xxx == xxx; // expected-warning {{self-comparison always evaluates to true}}
}
@end
