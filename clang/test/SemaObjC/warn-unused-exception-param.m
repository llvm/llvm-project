// RUN: %clang_cc1 -fsyntax-only -verify -fobjc-exceptions -Wunused-exception-parameter %s
void f0(void) {
  @try {} @catch(id a) {} // expected-warning{{unused exception parameter 'a'}}
}
