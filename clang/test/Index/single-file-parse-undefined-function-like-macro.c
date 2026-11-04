// RUN: split-file %s %t
// RUN: c-index-test -single-file-parse %t/tu.c 2>&1 | FileCheck %t/tu.c

//--- header.h
#define FUNCTION_LIKE_MACRO() 1
//--- tu.c
#include "header.h"
// CHECK-NOT: tu.c:[[@LINE+1]]:5: error: function-like macro 'FUNCTION_LIKE_MACRO' is not defined
#if FUNCTION_LIKE_MACRO()
// CHECK: tu.c:[[@LINE+1]]:5: FunctionDecl=then_fn
int then_fn();
#else
// CHECK: tu.c:[[@LINE+1]]:5: FunctionDecl=else_fn
int else_fn();
#endif
