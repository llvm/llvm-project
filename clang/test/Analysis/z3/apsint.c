// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux-gnu -analyzer-checker=core -verify %s
// expected-no-diagnostics

// https://bugs.llvm.org/show_bug.cgi?id=37622
_Bool a() {
  return !({ a(); });
}

// https://bugs.llvm.org/show_bug.cgi?id=37646
_Bool b;
void c() {
  _Bool a = b | 0;
  for (;;)
    if (a)
      ;
}
