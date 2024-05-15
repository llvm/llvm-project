// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.unix.BlockInCriticalSection -verify %s
// expected-no-diagnostics

// This should not crash
int (*a)(void);
void b(void) { a(); }
