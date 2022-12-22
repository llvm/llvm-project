void test() {
  for (auto [loopVar] : y) { // y has to be unresolved
    loopVa
  }
}
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-3):11 %s -o - \
// RUN:            | FileCheck %s
// CHECK: COMPLETION: loopVar
