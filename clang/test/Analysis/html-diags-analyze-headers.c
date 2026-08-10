// RUN: mkdir -p %t.dir
// RUN: %clang_analyze_cc1 -analyzer-opt-analyze-headers -analyzer-output=html -analyzer-checker=core -o %t.dir %s
// RUN: ls %t.dir | grep report
// RUN: rm -rf %t.dir

// This tests that we emit HTML diagnostics for reports in headers when the
// analyzer is run with -analyzer-opt-analyze-headers. This was handled
// incorrectly in the first iteration of D30406.

#include "html-diags-analyze-headers.h"
