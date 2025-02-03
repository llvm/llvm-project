void test() {
  if c
  // RUN: %clang_cc1 -fsyntax-only -std=c++17 -code-completion-at=%s:%(line-1):7 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: constexpr
