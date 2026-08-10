// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=optin.cplusplus.UninitializedObject \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

struct A {
  int *iptr;
  int a;  // expected-note{{uninitialized field 'this->a'}}
  int b;  // expected-note{{uninitialized field 'this->b'}}

  A (int *iptr) : iptr(iptr) {} // expected-warning{{2 uninitialized fields at the end of the constructor call [optin.cplusplus.UninitializedObject]}}
};

void f() {
  A a(0);
}

//CHECK:      <tr><td class="rowname">Note:</td>
//CHECK-NOT:  <a href="#Note0">
//CHECK-SAME: <a href="#Note1">line 9, column 7</a>
//CHECK-SAME: <a href="#Note2">line 10, column 7</a>
