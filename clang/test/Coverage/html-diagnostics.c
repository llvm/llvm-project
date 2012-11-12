// RUN: rm -rf %t
// RUN: %clang_cc1 -analyze -analyzer-output=html -analyzer-checker=core -o %t %s
// RUN: cat %t/*.html | FileCheck %s

// CHECK: <h3>Annotated Source Code</h3>

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


