// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl 2>&1 | FileCheck %s

// Test that [[nodiscard]] warnings are suppressed for REPL top-level
// expressions that will have their values printed (no semicolon),
// but are still emitted when the value is actually discarded (with semicolon).

extern "C" int printf(const char*,...);

[[nodiscard]] int getValue() { return 42; }

// Negative test: Warning when value is discarded (with semicolon)
getValue();
// CHECK: warning: ignoring return value of function declared with 'nodiscard' attribute

// Positive test: No warning when expression value is printed (no semicolon)
getValue()
// CHECK: (int) 42

// Verify assignment doesn't warn
int x = getValue();
printf("x = %d\n", x);
// CHECK: x = 42

%quit
