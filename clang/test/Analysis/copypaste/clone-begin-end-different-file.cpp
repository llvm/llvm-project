// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.clone.CloneChecker -analyzer-config alpha.clone.CloneChecker:MinimumCloneComplexity=5 -verify %s

// This test should verify that there is no crash if the detected clone range
// starts in a file and ends in a different file.

void f_end(int i) {
  if (i == 10) // expected-warning{{Duplicate code detected}}
#include "Inputs/clone-begin-end-different-file-end.inc"
  if (i == 10) // expected-note{{Similar code here}}
#include "Inputs/clone-begin-end-different-file-end.inc"
}

void f_begin(int i) {
#include "Inputs/clone-begin-end-different-file-begin-1.inc"
    if (true) {}
#include "Inputs/clone-begin-end-different-file-begin-2.inc"
    if (true) {}
}

#define X while (true) {}

void f1m(int i) {
  if (i == 10) // expected-warning{{Duplicate code detected}}
#include "Inputs/clone-begin-end-different-file-end-macro.inc"
  if (i == 10) // expected-note{{Similar code here}}
#include "Inputs/clone-begin-end-different-file-end-macro.inc"
}

#undef X
#define X if (i == 10)

void f2m(int i) {
#include "Inputs/clone-begin-end-different-file-begin-macro-1.inc"
    while (true) { i = 1; }
#include "Inputs/clone-begin-end-different-file-begin-macro-2.inc"
    while (true) { i = 1; }
}
