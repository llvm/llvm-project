// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -fsyntax-only -verify %s
// expected-no-diagnostics
// rdar://17259812

typedef void (^BT) ();

class S {
  BT br() __attribute__((ns_returns_retained)) {
    return ^{};
  }
 BT br1() __attribute__((ns_returns_retained));
};

BT S::br1() {
    return ^{};
}
