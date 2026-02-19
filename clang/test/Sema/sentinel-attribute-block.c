  // RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s

  // Test that sentinel attribute on block variables doesn't crash
  void foo(void) {
    void (^a)() __attribute__((__sentinel__)) = {}; // expected-warning {{'sentinel' attribute requires named arguments}}
  }
