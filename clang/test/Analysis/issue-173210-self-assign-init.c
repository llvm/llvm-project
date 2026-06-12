// RUN: %clang_analyze_cc1 -xc %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// RUN:   -verify
// RUN: %clang_analyze_cc1 -xc++ %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores \
// RUN:   -verify -w

// Self assignment initialization in C code will be treated as nop.
// We will report the VarDecl only if it was left uninitialized by the time of
// a subsequent DeclRefExpr.

// NOTE: No warnings from the deadcode.DeadStores checker.

void clang_analyzer_warnIfReached();

struct S { int x; };
union U { int x; };
enum T { TT };

// No need to test VarDecl of multiple variables, as they will be split into
// single ones when constructing the CFG.

int warnvar() {
  int x = x; // no-warnings for C/C++, binding is skipped via the
             // self-assignment filter.
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

int *warnptr() {
  int *p = p; // Same as warnvar.
  return p; // expected-warning{{Undefined or garbage value returned to caller}}
}

enum T warnenum() {
  enum T t = t; // Same as warnvar.
  return t; // expected-warning{{Undefined or garbage value returned to caller}}
}

int warnstruct() {
  struct S s = s; // no-warnings for C/C++
                  // In C, same as warnvar.
                  // In C++, binding is handled in the ctor call and s.x is
                  // bound to an Undefined.
  return s.x; // expected-warning{{Undefined or garbage value returned to caller}}
}

#ifndef __cplusplus
int warnunion() {
  union U u = u; // no-warnings for C/C++
                 // In C, same as warnvar.
                 // In C++, binding is handled in the ctor call and u is bound
                 // to a lazyCompoundVal, which will not trigger an undefined
                 // usage warning.
  return u.x; // expected-warning{{Undefined or garbage value returned to caller}}
}
#endif // not __cplusplus

// NOTE: The self assignment of reference type is also tested in stack-addr-ps.cpp.
// I.e., `int& i = i;` in function f5
// We only keep a simple regression confirmation here.
#ifdef __cplusplus
void warnref() {
  int &x = x; // expected-warning{{Assigned value is uninitialized}}
}
#endif // __cplusplus
