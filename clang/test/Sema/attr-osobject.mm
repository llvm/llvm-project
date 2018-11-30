// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-no-diagnostics

struct S {};

@interface I
  @property (readonly) S* prop __attribute__((os_returns_retained));
  - (S*) generateS __attribute__((os_returns_retained));
  - (void) takeS:(S*) __attribute__((os_consumed)) s;
@end
