// Note: the run lines follow their respective tests, since line/column
// matter in this test.
void f0(float x, float y);
void f1();
void test() {
  f0(0, 0);
  g0(0, 0);
  f1(0, 0);
  // RUN: %clang_cc1 -std=c89 -fsyntax-only  -code-completion-at=%s:%(line-3):6 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: f0(<#float x#>, float y)
  // RUN: %clang_cc1 -std=c89 -fsyntax-only -code-completion-at=%s:%(line-5):9 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
  // CHECK-CC2: f0(float x, <#float y#>)
  // RUN: %clang_cc1 -std=c89 -fsyntax-only -code-completion-at=%s:%(line-5):6 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
  // CHECK-CC3: f1()
}
