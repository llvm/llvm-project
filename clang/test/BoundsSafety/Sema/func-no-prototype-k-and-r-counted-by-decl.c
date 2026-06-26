// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fexperimental-bounds-safety-attributes -x c -verify %s
#include <ptrcheck.h>

// A K&R identifier-list declaration has no ParmVarDecl for its parameters, so
// late-parsing the return type's bounds attribute must tolerate a null
// parameter rather than crash.
int *__counted_by(n) foo(n);
// expected-error@-1{{use of undeclared identifier 'n'}}
// expected-error@-2{{a parameter list without types is only allowed in a function definition}}
