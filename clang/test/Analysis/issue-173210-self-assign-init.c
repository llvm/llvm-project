// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -verify

// Self assignment initialization in C code will be treated as nop.
// We will not report the VarDecl, but the following DeclRefExpr if it has not
// yet been initialized then.

void clang_analyzer_warnIfReached();

struct S { int x; };
union U { int x; };

void nowarn() {
  int x = x; // no-warning
  int *p = p; // no-warning
  struct S s = s; // no-warning
  union U u = u; // no-warning
  // Ensure the analysis is not terminated sliently.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

int warn() {
  int x = x;
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

// NOTE: The self assignment of reference type is tested with stack-addr-ps.cpp.
// I.e., `int& i = i;` in function f5
