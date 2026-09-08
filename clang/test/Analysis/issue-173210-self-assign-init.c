// RUN: %clang_analyze_cc1 %s -verify -xc \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores

// Suppress the C++ -Wuninitialized warnings on struct and reference.
// RUN: %clang_analyze_cc1 %s -verify -xc++ -Wno-uninitialized \
// RUN:   -analyzer-checker=core,debug.ExprInspection,deadcode.DeadStores

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

int check_var() {
  int x = x; // no-warnings for C/C++, binding is skipped via the
             // self-assignment filter.
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

int *check_ptr() {
  int *p = p; // Same as warnvar.
  return p; // expected-warning{{Undefined or garbage value returned to caller}}
}

enum T check_enum() {
  enum T t = t; // Same as warnvar.
  return t; // expected-warning{{Undefined or garbage value returned to caller}}
}

int check_struct() {
  // no-warnings for C/C++:
  // In C, same as warnvar.
  // In C++, binding is handled in the ctor call and 's.x' is bound to an Undefined.
  struct S s = s; // no-warnings
  return s.x; // expected-warning{{Undefined or garbage value returned to caller}}
}

#ifndef __cplusplus
int check_union() {
  // no-warnings for C/C++:
  // In C, same as warnvar.
  // In C++, binding is handled in the ctor call and 'u' is bound to a
  // lazyCompoundVal, which will not trigger an undefined usage warning.
  union U u = u; // no-warnings
  return u.x; // expected-warning{{Undefined or garbage value returned to caller}}
}
#endif // not __cplusplus

#ifdef __cplusplus

// NOTE: The self assignment of reference type is also tested in stack-addr-ps.cpp.
// I.e., `int& i = i;` in function f5
// We only keep a simple regression confirmation here.
void check_ref() {
  int &x = x; // expected-warning{{Assigned value is uninitialized}}
}

// Confirmation for default member initializer.
struct struct_self_assign {
  int x = x; // no-warnings
};
int check_struct_self_assign() {
  struct_self_assign s; // no-warnings
  return s.x; // FIXME: there should be a warning.
}

// Confirmation for constructor initialization list.
struct struct_init_list {
  int x;
  struct_init_list()
    : x(x) // expected-warning{{Assigned value is uninitialized}}
  {}
};
void check_struct_init_list() {
  // Trigger the ctor call.
  struct_init_list s;
}

// The below two have the same AST structure.
int check_copy_list_initialization() {
  int x = {x}; // no-warnings
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
};
int check_direct_list_initialization() {
  int x {x}; // no-warnings
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
};

// Having the same AST structure as `int x = x;`, rather than `int x = (x);`.
int check_direct_initialization() {
  int x (x); // no-warnings
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

#endif // __cplusplus

// Ignore parentheses for the initialization-with-a-macro cases, such as
// #define VAR_INIT(x) (x) and int x = VAR_INIT(x);
int check_paren_init() {
  int x = (x); // no-warnings
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}
