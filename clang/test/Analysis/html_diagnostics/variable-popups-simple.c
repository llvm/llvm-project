// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

void bar(int);

void foo2() {
  int a;
  int b = 1;
  if (b)
    bar(a); // expected-warning{{1st function call argument is an uninitialized value}}
}

// CHECK:      <span class='variable'>b
// CHECK-SAME:   <table class='variable_popup'><tbody><tr>
// CHECK-SAME:     <td valign='top'>
// CHECK-SAME:       <div class='PathIndex PathIndexPopUp'>1.1</div>
// CHECK-SAME:     </td>
// CHECK-SAME:     <td>'b' is 1</td>
// CHECK-SAME:   </tr></tbody></table>
// CHECK-SAME: </span>
