template <bool Flag>
void test() {
  if constexpr (Flag) {
    return;
  }
  // RUN: %clang_cc1 -fsyntax-only -std=c++17 -code-completion-at=%s:3:7 %s -o - | FileCheck -check-prefix=CHECK-CXX17 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-at=%s:3:7 %s -o - | FileCheck -check-prefix=CHECK-CXX23 %s
  // CHECK-CXX17: constexpr
  // CHECK-CXX23: consteval
  if !c
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-at=%s:10:8 %s -o - | FileCheck -check-prefix=CHECK-CXX23-NOT %s
  // CHECK-CXX23-NOT: consteval
}
