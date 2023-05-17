enum X { x };
enum Y { y };
struct Z { };

void X(void);

void test(void) {
  enum X { x };
  enum 
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):8 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: X
  // CHECK-CC1: Y
