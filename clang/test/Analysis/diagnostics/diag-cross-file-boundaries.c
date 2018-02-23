// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=html -o PR12421.html %s 2>&1 | FileCheck %s

// Test for PR12421
#include "diag-cross-file-boundaries.h"

int main(){
  f();
  return 0;
}

// CHECK: warning: Path diagnostic report is not generated.
