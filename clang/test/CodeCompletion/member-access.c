struct Point {
  float x;
  float y;
  float z;
};

void test(struct Point *p) {
  p->
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:8:6 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
  // CHECK-CC1: x
  // CHECK-CC1: y
  // CHECK-CC1: z
}

struct Point2 {
  float x;
};

void test2(struct Point2 p) {
  p->
}

void test3(struct Point2 *p) {
  p.
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:20:6 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: x (requires fix-it: {20:4-20:6} to ".")

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:24:5 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: x (requires fix-it: {24:4-24:5} to "->")
