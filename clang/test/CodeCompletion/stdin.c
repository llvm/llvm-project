enum X { x };
enum Y { y };

enum 
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=-:4:6 < %s -o - | FileCheck -check-prefix=CC1 %s
  // CHECK-CC1: X
  // CHECK-CC1: Y
