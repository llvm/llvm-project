#define ID(X) X

void test(bool input_var) {
  ID(input_var) = true;
  // Check that input_var shows up when completing at the start, in the middle
  // and at the end of the identifier.
  //
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:6 %s -o - | FileCheck %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:8 %s -o - | FileCheck %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:4:15 %s -o - | FileCheck %s

  // CHECK: input_var
}
