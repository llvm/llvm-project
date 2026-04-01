// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:   -triple x86_64-pc-linux-gnu \
// RUN:   -verify %s
// The bit-cast dereference produces a LazyCompoundVal which must not crash when handled
// by the constraint manager.

void foo() {
  char arr[0];
  if (*__builtin_bit_cast(int *, &arr)) // expected-warning {{Branch condition evaluates to a garbage value}}
    ;
}
