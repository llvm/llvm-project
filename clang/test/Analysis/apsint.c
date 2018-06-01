// REQUIRES: z3
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -analyzer-checker=core -verify %s
// expected-no-diagnostics

_Bool a() {
  return !({ a(); });
}
