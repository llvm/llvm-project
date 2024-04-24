// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: grep -v CHECK %t/report-*.html | FileCheck %s


void foo() {
  int *x = 0;
  *x = __COUNTER__; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

void bar() {
  int *y;
  *y = __COUNTER__; // expected-warning{{Dereference of undefined pointer value (loaded from variable 'y')}}
}

// The checks below confirm that both reports have the same values for __COUNTER__.
//
// FIXME: The correct values are (0, 1, 0, 1). Because we re-lex the file in order
// to detect macro expansions for HTML report purposes, they turn into (2, 3, 2, 3)
// by the time we emit HTML. But at least it's better than (2, 3, 4, 5)
// which would have been the case if we re-lexed the file *each* time we
// emitted an HTML report.

// CHECK: <span class='macro'>__COUNTER__<span class='macro_popup'>2</span></span>
// CHECK: <span class='macro'>__COUNTER__<span class='macro_popup'>3</span></span>
// CHECK: <span class='macro'>__COUNTER__<span class='macro_popup'>2</span></span>
// CHECK: <span class='macro'>__COUNTER__<span class='macro_popup'>3</span></span>
