// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s
// REQUIRES: asserts

void escapes() {
  // CHECK: <td align="left"><b>Store: </b> <font color="gray">(0x{{[0-9a-f]*}})</font></td>
  // CHECK-SAME: <td align="left">foo</td><td align="left">0</td>
  // CHECK-SAME: <td align="left">&amp;Element\{"foo",0 S64b,char\}</td>
  // CHECK: <td align="left"><b>Expressions: </b></td>
  // CHECK-SAME: <td align="left">"foo"</td>
  // CHECK-SAME: <td align="left">&amp;Element\{"foo",0 S64b,char\}</td>
  const char *const foo = "\x66\x6f\x6f";

  // CHECK: <font color="cyan4">BinaryOperator</font>
  // CHECK-SAME: <td align="left">1 \| 2</td>
  // CHECK-SAME: <td align="left">3 S32b</td>
  int x = 1 | 2;
}
