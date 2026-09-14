// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

void bar(int);

void foo(void) {
  int a;
  int b = 1;
  if (b
      && 1)
    bar(a); // expected-warning{{1st function call argument is an uninitialized value}}
}

// The variable 'b' is the last token on its line, so the popup's closing
// tags and the row's closing </td></tr> are inserted at the same offset.
// The popup table must immediately follow the variable, and the closing
// </span> tags (variable and control-flow arrow anchors) must stay inside
// the row, before </td></tr>. Each directive below matches contiguously.
// CHECK:      <span class='variable'>b<table class='variable_popup'><tbody>
// CHECK-SAME: <tr><td valign='top'><div class='PathIndex PathIndexPopUp'>1.1</div></td><td>'b' is 1</td></tr>
// CHECK-SAME: </tbody></table></span></span></span></td></tr>
