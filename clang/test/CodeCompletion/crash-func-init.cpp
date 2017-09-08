int (*foo(int a))(flo
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:1:21 %s -o - \
// RUN:            | FileCheck %s
// CHECK: COMPLETION: float
