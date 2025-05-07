

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

void has_no_prototype();

void call_has_no_prototype(int *__counted_by(count) p, int count) {
  int *a = p;
  has_no_prototype(a);
}

// CHECK:|-FunctionDecl {{.*}} has_no_prototype 'void ()'

// CHECK-LABEL:`-FunctionDecl {{.*}} call_has_no_prototype 'void (int *__single __counted_by(count), int)'
// CHECK:        `-CallExpr {{.*}} 'void'
// CHECK-NEXT:      |-ImplicitCastExpr {{.*}} 'void (*__single)()' <FunctionToPointerDecay>
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'void ()' Function {{.*}} 'has_no_prototype' 'void ()'
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:          `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'a' 'int *__bidi_indexable'
