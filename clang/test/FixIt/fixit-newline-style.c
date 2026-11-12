// RUN: %clang_cc1 -pedantic -Wunused-label -fno-diagnostics-show-line-numbers -x c %s 2>&1 | FileCheck %s -strict-whitespace

// This file intentionally uses a CRLF newline style
// CHECK: warning: unused label 'ddd'
// CHECK-NEXT: {{^  ddd:}}
// CHECK-NEXT: {{^  \^~~~$}}
// CHECK-NOT: {{^  ;}}
void f(void) {
  ddd:
  ;
}
