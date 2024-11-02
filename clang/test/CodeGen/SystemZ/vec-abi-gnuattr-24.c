// RUN: %clang_cc1 -triple s390x-ibm-linux -emit-llvm -fzvector -o - %s 2>&1 \
// RUN:   | FileCheck  %s
//
// Test that the "s390x-visible-vector-ABI" module flag is not emitted.

// Use of va_arg with a scalar type.
#include <stdarg.h>
int fun0(va_list vl) {
  return va_arg(vl, int);
}

typedef __attribute__((vector_size(16))) int v4i32;

// Declaring unused global function with vector argument and return values;
v4i32 globfun(v4i32 Arg);

// Declaring global scalar variable used below.
int GlobVal;

// Declaring extern global scalar variable used below.
extern int GlobExtVar;

// Local vector variable used below.
static v4i32 Var;

// Local function with vector argument and return values;
static v4i32 foo(v4i32 Arg) {
  Var = Var + Arg;
  return Var;
}

int fun1() {
  v4i32 V = {1, 2, 3, 4};
  return foo(V)[0] + GlobVal + GlobExtVar;
}

//CHECK-NOT: !{i32 2, !"s390x-visible-vector-ABI", i32 1}
