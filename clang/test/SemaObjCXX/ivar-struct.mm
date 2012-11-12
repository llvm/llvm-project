// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
@interface A {
  struct X {
    int x, y;
  } X;
}
@end
