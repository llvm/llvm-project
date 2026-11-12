class foo {
  void mut_func() {
    [this]() {

    }();
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):1 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
    // CHECK-CC1: const_func
    // CHECK-CC1: mut_func
  }

  void const_func() const {
    [this]() {

    }();
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-2):1 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
    // CHECK-CC2-NOT: mut_func
    // CHECK-CC2: const_func
  };
};


