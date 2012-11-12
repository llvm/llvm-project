class T { };

typedef int Integer;

namespace N { }

void f() {
  typedef float Float;
  
  operator
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:10:11 %s -o - | FileCheck -check-prefix=CC1 %s
  // CHECK-CC1: +
  // CHECK-CC1: Float
  // CHECK-CC1: Integer
  // CHECK-CC1: N
  // CHECK-CC1: short
  // CHECK-CC1: T
