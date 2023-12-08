// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

void bar(int);

void foo(void) {
  int a;
  for (unsigned i = 0; i < 3; ++i)
    if (i)
      bar(a); // expected-warning{{1st function call argument is an uninitialized value}}
}

// CHECK:      <span class='variable'>i
// CHECK-SAME:   <table class='variable_popup'><tbody><tr>
// CHECK-SAME:     <td valign='top'>
// CHECK-SAME:       <div class='PathIndex PathIndexPopUp'>2.1</div>
// CHECK-SAME:     </td>
// CHECK-SAME:     <td>'i' is 0</td>
// CHECK-SAME:   </tr>
// CHECK-SAME:   <tr>
// CHECK-SAME:     <td valign='top'>
// CHECK-SAME:       <div class='PathIndex PathIndexPopUp'>4.1</div>
// CHECK-SAME:     </td>
// CHECK-SAME:     <td>'i' is 1</td>
// CHECK-SAME:   </tr></tbody></table>
// CHECK-SAME: </span>
