

// RUN: %clang_cc1 -ast-dump -fbounds-safety -Wno-error=implicit-function-declaration %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -Wno-error=implicit-function-declaration -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

void call_undeclared_function(int *__counted_by(count) p, int count) {
  int *a = p;
  undeclared_function(a);
}
// CHECK:{{.*}}warning: call to undeclared function 'undeclared_function'; ISO C99 and later do not support implicit function declarations

// CHECK-LABEL:`-FunctionDecl {{.*}} call_undeclared_function 'void (int *__single __counted_by(count), int)'
// CHECK:    `-CallExpr {{.*}} 'int'
// CHECK-NEXT:      |-ImplicitCastExpr {{.*}} 'int (*__single)()' <FunctionToPointerDecay>
// CHECK-NEXT:      | `-DeclRefExpr {{.*}} 'int ()' Function {{.*}} 'undeclared_function' 'int ()'
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:          `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'a' 'int *__bidi_indexable'
