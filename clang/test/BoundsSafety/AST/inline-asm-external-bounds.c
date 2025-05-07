

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

void Test() {
  __asm__ ("test %0" ::"r" (("beef")));
}
// CHECK-LABEL: Test 'void ()'
// CHECK-NEXT:  `-CompoundStmt {{.*}}
// CHECK-NEXT:    `-GCCAsmStmt {{.*}}
// CHECK-NEXT:      `-ImplicitCastExpr {{.*}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:        `-ImplicitCastExpr {{.*}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:          `-ParenExpr {{.*}} 'char[5]' lvalue
// CHECK-NEXT:            `-StringLiteral {{.*}} 'char[5]' lvalue "beef"
