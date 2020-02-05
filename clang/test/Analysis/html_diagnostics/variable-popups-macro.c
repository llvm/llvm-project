// RUN: rm -fR %t
// RUN: mkdir %t
// RUN: %clang_analyze_cc1 -analyzer-checker=core \
// RUN:                    -analyzer-output=html -o %t -verify %s
// RUN: cat %t/report-*.html | FileCheck %s

void bar(int);

#define MACRO if (b)

void foo2() {
  int a;
  int b = 1;
  MACRO
    bar(a); // expected-warning{{1st function call argument is an uninitialized value}}
}

// For now we don't emit popups inside macros due to UI limitations.
// Once we do, we should test it thoroughly.

// CHECK-LABEL: <tr class="codeline" data-linenumber="14">
// CHECK-NOT:   <span class='variable'>
// CHECK-SAME:  <span class='macro'>
// CHECK-SAME:    MACRO
// CHECK-SAME:    <span class='macro_popup'>
// CHECK-SAME:      if (b)
// CHECK-SAME:    </span>
// CHECK-SAME:  </span>
