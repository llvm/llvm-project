// Note: the run lines follow their respective tests, since line/column
// matter in this test.

void func() {
  int *foo = (int *)(0x200);
  int *bar = (int *)((0x200));
}

// Make sure this doesn't crash
// RUN: %clang_cc1 -fsyntax-only -xobjective-c++-header -code-completion-at=%s:%(line-5):28 %s
// RUN: %clang_cc1 -fsyntax-only -xobjective-c++-header -code-completion-at=%s:%(line-5):30 %s

