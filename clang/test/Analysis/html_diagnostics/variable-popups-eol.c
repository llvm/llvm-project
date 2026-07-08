// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=html -o %t -verify %s
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
// The popup table and the closing </span> must stay inside the row.
// CHECK:      <span class='variable'>b
// CHECK-SAME:   <table class='variable_popup'><tbody>
// CHECK-SAME:     <tr><td valign='top'>
// CHECK-SAME:       <div class='PathIndex PathIndexPopUp'>1.1</div>
// CHECK-SAME:     </td><td>'b' is 1</td></tr>
// CHECK-SAME:   </tbody></table>
// CHECK-SAME: </span>
// CHECK-SAME: </td></tr>
