// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks -fobjc-arc
// RUN: %clang_cc1 -fsyntax-only %s -verify -fblocks

__auto_type block = ^ id __attribute__((ns_returns_retained)) (id filter) {
  return filter; // ok
};
__auto_type block2 = ^  __attribute__((ns_returns_retained)) id (id filter) {
  return filter; // ok
};
__auto_type block3 = ^ id (id filter)  __attribute__((ns_returns_retained))  {
  return filter; // ok
};

// expected-no-diagnostics
