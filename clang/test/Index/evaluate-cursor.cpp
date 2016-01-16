// Test is line- and column-sensitive. Run lines are below.

struct Foo {
  int x = 10;
};

void foo() {
  int p = 11;
}

#define FUNC_MAC(x) x

void goo() {
  int p = FUNC_MAC(1);
  int a = __LINE__;
}

// RUN: c-index-test -evaluate-cursor-at=%s:4:7 \
// RUN:    -evaluate-cursor-at=%s:8:7 \
// RUN:    -evaluate-cursor-at=%s:8:11 -std=c++11 %s | FileCheck %s
// CHECK: Value: 10
// CHECK: Value: 11
// CHECK: Value: 11

// RUN: c-index-test -get-macro-info-cursor-at=%s:11:9 \
// RUN:    -get-macro-info-cursor-at=%s:14:11 \
// RUN:    -get-macro-info-cursor-at=%s:15:11 -std=c++11 %s | FileCheck -check-prefix=CHECK-MACRO %s
// CHECK-MACRO: [function macro]
// CHECK-MACRO: [function macro]
// CHECK-MACRO: [builtin macro]
