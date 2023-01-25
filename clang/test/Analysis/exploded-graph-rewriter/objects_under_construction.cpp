// FIXME: Figure out how to use %clang_analyze_cc1 with our lit.local.cfg.
// RUN: %clang_cc1 -std=c++14 -analyze -triple x86_64-unknown-linux-gnu \
// RUN:                     -analyze-function "test()" \
// RUN:                     -analyzer-checker=core \
// RUN:                     -analyzer-dump-egraph=%t.dot %s
// RUN: %exploded_graph_rewriter %t.dot | FileCheck %s

struct S {
  S() {}
};

void test() {
  // CHECK: Objects Under Construction:
  // CHECK-SAME: <tr>
  // CHECK-SAME:   <td align="left"><b>#0 Call</b></td>
  // CHECK-SAME:   <td align="left" colspan="2">
  // CHECK-SAME:     <font color="gray60">test </font>
  // CHECK-SAME:   </td>
  // CHECK-SAME: </tr>
  // CHECK-SAME: <tr>
  // CHECK-SAME:   <td align="left"><i>S{{[0-9]*}}</i></td>
  // CHECK-SAME:   <td align="left"><font color="darkgreen"><i>
  // CHECK-SAME:     (materialize temporary)
  // CHECK-SAME:   </i></font></td>
  // CHECK-SAME:   <td align="left">S()</td>
  // CHECK-SAME:   <td align="left">&amp;s</td>
  // CHECK-SAME: </tr>
  // CHECK-SAME: <tr>
  // CHECK-SAME:   <td align="left"><i>S{{[0-9]*}}</i></td>
  // CHECK-SAME:   <td align="left"><font color="darkgreen"><i>
  // CHECK-SAME:     (elide constructor)
  // CHECK-SAME:   </i></font></td>
  // CHECK-SAME:   <td align="left">S()</td>
  // CHECK-SAME:   <td align="left">&amp;s</td>
  // CHECK-SAME: </tr>
  // CHECK-SAME: <tr>
  // CHECK-SAME:   <td align="left"><i>S{{[0-9]*}}</i></td>
  // CHECK-SAME:   <td align="left"><font color="darkgreen"><i>
  // CHECK-SAME:     (construct into local variable)
  // CHECK-SAME:   </i></font></td>
  // CHECK-SAME:   <td align="left">S s = S();</td>
  // CHECK-SAME:   <td align="left">&amp;s</td>
  // CHECK-SAME: </tr>
  S s = S();
}
