// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-checker=unix.Malloc -analyzer-checker=unix.Stream \
// RUN:   -verify %s

// Several checkers ("partially modeled" calls — MallocChecker, StreamChecker,
// CStringChecker, etc.) hand off to ProgramState::invalidateRegions for the
// fallback parts of their modeling. After this patch the resulting symbols
// carry a "partial-call" cause to distinguish them from values produced by a
// pure conservative-eval call.

#include "Inputs/system-header-simulator-for-simple-stream.h"

void clang_analyzer_dump_int(int);
void clang_analyzer_dump_ptr(int *);

// ----- StreamChecker: fread invalidates the buffer with a partial-call cause.
void test_fread_buffer_invalidation(void) {
  FILE *F = fopen("/tmp/x", "r");
  if (!F)
    return;
  int buf[2] = {0, 0};
  fread(buf, sizeof(int), 2, F);
  // The buffer's default binding is now an inv_$ artifact carrying a
  // "partial-call" cause.
  clang_analyzer_dump_int(buf[0]); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, partial-call, S[0-9]+, #[0-9]+}}}}}
  fclose(F);
}
