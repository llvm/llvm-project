// RUN: rm -rf %t
// RUN: %clang_cc1 -analyze -analyzer-output=html -analyzer-checker=core -o %t %s
// RUN: find %t -name "*.html" -exec cat "{}" ";" | FileCheck %s
//
// RUN: rm -rf %t
// RUN: %clang_cc1 -analyze -analyzer-output=html-single-file -analyzer-checker=core -o %t %s
// RUN: find %t -name "*.html" -exec cat "{}" ";" | FileCheck %s

// REQUIRES: staticanalyzer

// CHECK: <h3>Annotated Source Code</h3>

// Make sure it's not generated as a multi-file HTML output
// CHECK-NOT: <h4 class=FileName>{{.*}}

// Without tweaking expr, the expr would hit to the line below
// emitted to the output as comment.
// CHECK: {{[D]ereference of null pointer}}

void f0(int x) {
  int *p = &x;

  if (x > 10) {
    if (x == 22)
      p = 0;
  }

  *p = 10;
}


