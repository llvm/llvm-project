// Check the case when only the StreamChecker is enabled.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,unix.Stream \
// RUN:   -analyzer-config unix.Stream:Pedantic=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple x86_64-unknown-linux \
// RUN:   -verify=stream

// Check the case when only the StdLibraryFunctionsChecker is enabled.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple x86_64-unknown-linux \
// RUN:   -verify=stdLib 2>&1 | FileCheck %s

// Check the case when both the StreamChecker and the
// StdLibraryFunctionsChecker are enabled.
// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,unix.Stream \
// RUN:   -analyzer-config unix.Stream:Pedantic=true \
// RUN:   -analyzer-checker=unix.StdCLibraryFunctions \
// RUN:   -analyzer-config unix.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple x86_64-unknown-linux \
// RUN:   -verify=both 2>&1 | FileCheck %s

// Verify that the summaries are loaded when the StdLibraryFunctionsChecker is
// enabled.
//      CHECK: Loaded summary for: int getchar(void)
// CHECK-NEXT: Loaded summary for: unsigned long fread(void *restrict, size_t, size_t, FILE *restrict)
// CHECK-NEXT: Loaded summary for: unsigned long fwrite(const void *restrict, size_t, size_t, FILE *restrict)

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);

void test_fread_fwrite(FILE *fp, int *buf) {
  fp = fopen("foo", "r");
  if (!fp)
    return;
  size_t x = fwrite(buf, sizeof(int), 10, fp);

  clang_analyzer_eval(x <= 10); // \
 // stream-warning{{TRUE}} \
 // stdLib-warning{{TRUE}} \
 // both-warning{{TRUE}}

  clang_analyzer_eval(x == 10); // \
  // stream-warning{{TRUE}} \
  // stream-warning{{FALSE}} \
  // stdLib-warning{{TRUE}} \
  // stdLib-warning{{FALSE}} \
  // both-warning{{TRUE}} \
  // both-warning{{FALSE}}

  fclose(fp);
}
