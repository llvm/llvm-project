struct Point {
  float x;
  float y;
  float z;
};

void test(struct Point *p) {
  p->
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):6 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
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

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:%(line-7):6 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: x (requires fix-it: {[[@LINE-8]]:4-[[@LINE-8]]:6} to ".")

// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:%(line-6):5 %s -o - | FileCheck -check-prefix=CHECK-CC3 %s
// CHECK-CC3: x (requires fix-it: {[[@LINE-7]]:4-[[@LINE-7]]:5} to "->")

void test4(struct Point *p) {
  (int)(p)->x;
  (int)(0,1,2,3,4,p)->x;
}
// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:%(line-3):13 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
// RUN: %clang_cc1 -fsyntax-only -code-completion-with-fixits -code-completion-at=%s:%(line-3):23 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
