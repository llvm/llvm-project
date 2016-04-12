// RUN: clang-tidy %s -checks="-*,misc-suspicious-semicolon" -- 2>&1 | FileCheck %s

// Note: This test verifies that, the checker does not emit any warning for
//       files that do not compile.

bool g();

void f() {
  if (g());
  // CHECK-NOT: [misc-suspicious-semicolon]
  int a
}
