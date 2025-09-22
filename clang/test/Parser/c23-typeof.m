// RUN: %clang_cc1 -verify -std=c23 -fblocks -Wno-unused %s
// expected-no-diagnostics

void f() {
  ^ typeof((void)0) {}; // Ok
  ^ typeof(void) {};    // Ok
}
