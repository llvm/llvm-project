struct Point {
  float x;
  float y;
  float z;
};

void test(struct Point *p) {
  p->
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:8:6 %s -o - | FileCheck -check-prefix=CC1 %s
  // CHECK-CC1: x
  // CHECK-CC1: y
  // CHECK-CC1: z
