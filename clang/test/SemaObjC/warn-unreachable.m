// RUN: %clang %s -fsyntax-only -Xclang -verify -fblocks -Wunreachable-code -Wno-unused-value -Wno-covered-switch-default

// This previously triggered a warning from -Wunreachable-code because of
// a busted CFG.
typedef signed char BOOL;
BOOL radar10989084() {
  @autoreleasepool {  // no-warning
    return __objc_yes;
  }
}

// Test the warning works.
void test_unreachable() {
  return;
  return; // expected-warning {{will never be executed}}
}

