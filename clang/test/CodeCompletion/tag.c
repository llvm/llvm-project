enum X { x };
enum Y { y };
struct Z { };

void X();

void test() {
  enum X { x };
  enum
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:9:7 %s -o - | FileCheck -check-prefix=CC1 %s
  // CHECK-CC1: X
  // CHECK-CC1: Y
