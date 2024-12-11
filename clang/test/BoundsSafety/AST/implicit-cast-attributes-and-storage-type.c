

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

extern int *__single extern_ptr;

void foo(void) {
  // Silence these expected warnings so they don't show up in AST dump.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
#pragma clang diagnostic ignored "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"
  char *__bidi_indexable b = extern_ptr;
#pragma clang diagnostic pop
}

// This test exists to make sure that the implementation of the
// "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"
// warning doesn't break the requirement that the above implicit conversion is
// split into a BitCast then BoundsSafetyPointerCast.

// CHECK:|-VarDecl {{.*}} used extern_ptr 'int *__single' extern
// CHECK-NEXT:`-FunctionDecl {{.*}} foo 'void (void)'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    `-DeclStmt {{.*}}
// CHECK-NEXT:      `-VarDecl {{.*}} b 'char *__bidi_indexable' cinit

// Make sure the `(int* __single) -> (char* __bidi_indexable) gets split into two
// separate implicit casts.
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:          `-ImplicitCastExpr {{.*}} 'char *__single' <BitCast>

// CHECK-NEXT:            `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
// CHECK-NEXT:              `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'extern_ptr' 'int *__single'
