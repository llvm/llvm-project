// RUN: %clang_analyze_cc1 -analyzer-checker=core -fblocks -verify %s

// expected-no-diagnostics

namespace block_rvo_crash {
struct A {};

A getA();
void use(A a) {}

void foo() {
  // This used to crash when finding construction context for getA()
  // (which is use()'s argument due to RVO).
  use(^{
    return getA();  // no-crash
  }());
}
} // namespace block_rvo_crash
