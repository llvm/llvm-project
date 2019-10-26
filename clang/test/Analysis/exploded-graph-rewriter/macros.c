// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s
// REQUIRES: asserts

// FIXME: Substitution doesn't seem to work on Windows.
// UNSUPPORTED: system-windows

// CHECK: macros.c:<b>17</b>:<b>10</b>
// CHECK-SAME: <font color="royalblue1">
// CHECK-SAME:   (<i>spelling at </i> macros.c:<b>15</b>:<b>14</b>)
// CHECK-SAME: </font>
#define NULL 0
void *foo() {
  return NULL;
}
