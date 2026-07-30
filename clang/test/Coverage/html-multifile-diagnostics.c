// RUN: rm -rf %t
// RUN: %clang_cc1 -analyze -analyzer-output=html -analyzer-checker=core -o %t %s
// RUN: find %t -name "*.html" -exec cat "{}" ";" | FileCheck %s

// REQUIRES: staticanalyzer

// CHECK: <!-- FILENAME html-multifile-diagnostics.h -->

// CHECK: <h3>Annotated Source Code</h3>

// Make sure it's generated as multi-file HTML output
// CHECK: <h4 class=FileName>{{.*}}html-multifile-diagnostics.c</h4>
// CHECK: <h4 class=FileName>{{.*}}html-multifile-diagnostics.h</h4>

// Without tweaking expr, the expr would hit to the line below
// emitted to the output as comment.
// CHECK: {{[D]ereference of null pointer}}

#include "html-multifile-diagnostics.h"

void f0(void) {
  f1((int*)0);
}
