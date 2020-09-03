// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s
// REQUIRES: asserts

struct A {
  A() {}
};

struct B {
  A a;
  B() : a() {}
};

void test() {
  // CHECK: (construct into member variable)
  // CHECK-SAME: <td align="left">a</td>
  // CHECK-SAME: <td align="left">&amp;b.a</td>
  B b;
}
