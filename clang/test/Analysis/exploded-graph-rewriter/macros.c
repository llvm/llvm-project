#define NULL 0
void *foo(void) {
  return NULL;
}

// The code above shall go first, because check tags below are sensetive to the line numbers on which the code is placed.
// You can change lines below in the way you need.

// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s

// CHECK: macros.c:<b>3</b>:<b>10</b>
// CHECK-SAME: <font color="royalblue1">
// CHECK-SAME:   (<i>spelling at </i> macros.c:<b>1</b>:<b>14</b>)
// CHECK-SAME: </font>
