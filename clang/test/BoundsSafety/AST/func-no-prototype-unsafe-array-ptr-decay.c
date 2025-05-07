

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

void has_no_prototype();

void call_has_no_prototype() {
  int buffer[5] = {0};
  return has_no_prototype(buffer);
}

// CHECK-LABEL:|-FunctionDecl {{.*}} used has_no_prototype 'void ()'
// CHECK-LABEL:`-FunctionDecl {{.*}} call_has_no_prototype 'void ()'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    |-DeclStmt {{.*}}
// CHECK-NEXT:    | `-VarDecl {{.*}} used buffer 'int[5]' cinit
// ...
// CHECK:          `-ReturnStmt {{.*}}
// CHECK-NEXT:      `-CallExpr {{.*}} 'void'
// CHECK-NEXT:        |-ImplicitCastExpr {{.*}} 'void (*__single)()' <FunctionToPointerDecay>
// CHECK-NEXT:        | `-DeclRefExpr {{.*}} 'void ()' Function {{.*}} 'has_no_prototype' 'void ()'
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:          `-ImplicitCastExpr {{.*}}'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:            `-DeclRefExpr {{.*}} 'int[5]' lvalue Var {{.*}} 'buffer' 'int[5]'
