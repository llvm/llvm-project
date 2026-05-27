// RUN: %clang_analyze_cc1 -xc %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// RUN:   -verify
// RUN: %clang_analyze_cc1 -xc++ %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// RUN:   -verify -w

// Self assignment initialization in C code will be treated as nop.
// We will not report the VarDecl, but the following DeclRefExpr if it has not
// yet been initialized then.

void clang_analyzer_warnIfReached();

struct S { int x; };
union U { int x; };

void nowarn() {
  int x = x; // no-warnings for C/C++
  int *p = p; // no-warnings for C/C++
  struct S s = s; // no-warning for C, but C++ will not report
  union U u = u; // no-warning for C, but C++ will not report
  // Ensure the analysis is not terminated sliently.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

int warn() {
  int x = x; // no-warnings for C/C++
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

// NOTE: The self assignment of reference type is also tested in stack-addr-ps.cpp.
// E.g., `int& i = i;` in function f5
// We only keep a simple regression confirmation here.
#ifdef __cplusplus
void warnref() {
  int &x = x; // expected-warning{{Assigned value is uninitialized}}
}
#endif // __cplusplus
