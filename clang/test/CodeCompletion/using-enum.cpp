enum class AAA { X, Y, Z };

namespace N2 {
  using enum AAA;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:14 %s | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: AAA
};
