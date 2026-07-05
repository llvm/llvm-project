struct A {
  A() = default;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):9 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: COMPLETION: default
  // CHECK-CC1-NEXT: COMPLETION: delete

  A(const A &) = default;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):18 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: COMPLETION: default
  // CHECK-CC2-NEXT: COMPLETION: delete

  A(const A &, int) = delete;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):23 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3-NOT: COMPLETION: default
  // CHECK-CC3: COMPLETION: delete

  A(A &&);

  A &operator=(const A &) = default;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):29 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC4 %s
  // CHECK-CC4: COMPLETION: default
  // CHECK-CC4-NEXT: COMPLETION: delete

  bool operator==(const A &) const = delete;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):38 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC5 %s
  // CHECK-CC5-NOT: COMPLETION: default
  // CHECK-CC5: COMPLETION: delete

  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-5):38 -std=gnu++20 %s -o - | FileCheck -check-prefix=CHECK-CC6 %s
  // CHECK-CC6: COMPLETION: default
  // CHECK-CC6-NEXT: COMPLETION: delete

  void test() = delete;
  // RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):17 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC7 %s
  // CHECK-CC7-NOT: COMPLETION: default
  // CHECK-CC7: COMPLETION: delete
};

A::A(A &&) = default;
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):14 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC8 %s
// CHECK-CC8: COMPLETION: default
// CHECK-CC8-NEXT: COMPLETION: delete

void test() = delete;
// RUN: %clang_cc1 -fsyntax-only -code-completion-patterns -code-completion-at=%s:%(line-1):15 -std=gnu++11 %s -o - | FileCheck -check-prefix=CHECK-CC9 %s
// CHECK-CC9-NOT: COMPLETION: default
// CHECK-CC9: COMPLETION: delete