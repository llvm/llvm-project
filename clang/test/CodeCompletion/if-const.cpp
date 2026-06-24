template <bool Flag>
void test() {
  if constexpr (Flag) {
    return;
  }
  // RUN: %clang_cc1 -fsyntax-only -std=c++17 -code-completion-at=%s:3:7 %s | FileCheck -check-prefix=CHECK-CXX17 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++17 -code-completion-patterns -code-completion-at=%s:3:7 %s | FileCheck -check-prefix=CHECK-PATTERN-CXX17 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-at=%s:3:7 %s | FileCheck -check-prefix=CHECK-CXX23 %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-patterns -code-completion-at=%s:3:7 %s | FileCheck -check-prefix=CHECK-PATTERN-CXX23 %s
  // CHECK-CXX17: COMPLETION: constexpr
  // CHECK-PATTERN-CXX17: COMPLETION: Pattern : constexpr (<#condition#>) {
  // CHECK-PATTERN-CXX17: <#statements#>
  // CHECK-PATTERN-CXX17: }
  // CHECK-CXX23: COMPLETION: consteval
  // CHECK-CXX23: COMPLETION: constexpr
  // CHECK-PATTERN-CXX23: COMPLETION: Pattern : consteval {
  // CHECK-PATTERN-CXX23: <#statements#>
  // CHECK-PATTERN-CXX23: }
  // CHECK-PATTERN-CXX23: COMPLETION: Pattern : constexpr (<#condition#>) {
  // CHECK-PATTERN-CXX23: <#statements#>
  // CHECK-PATTERN-CXX23: }
  if !c
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-at=%s:22:8 %s -o - | FileCheck -check-prefix=CHECK-CXX23-EXCLAIM %s
  // RUN: %clang_cc1 -fsyntax-only -std=c++23 -code-completion-patterns -code-completion-at=%s:22:8 %s -o - | FileCheck -check-prefix=CHECK-PATTERN-CXX23-EXCLAIM %s
  // CHECK-CXX23-EXCLAIM: COMPLETION: consteval
  // CHECK-CXX23-EXCLAIM-NOT: constexpr
  // CHECK-PATTERN-CXX23-EXCLAIM: COMPLETION: Pattern : consteval {
  // CHECK-PATTERN-CXX23-EXCLAIM: <#statements#>
  // CHECK-PATTERN-CXX23-EXCLAIM: }
}
