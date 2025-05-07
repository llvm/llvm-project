
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s | FileCheck %s

#include <ptrcheck.h>

#define NULL ((void *__single)0)

// CHECK-LABEL: test_null_to_bidi 'void ()'
// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl {{.*}} impl_bidi_ptr 'int *__bidi_indexable' cinit
// CHECK: |   `-CStyleCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__single' <BitCast> part_of_explicit_cast
// CHECK: |       `-ParenExpr {{.*}} 'void *__single'
// CHECK: |         `-CStyleCastExpr {{.*}} 'void *__single' <NullToPointer>
// CHECK: |           `-IntegerLiteral {{.*}} 'int' 0
// CHECK: `-DeclStmt
// CHECK:   `-VarDecl {{.*}} impl_bidi_ptr2 'int *__bidi_indexable' cinit
// CHECK:     `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       `-ImplicitCastExpr {{.*}} 'int *__single' <BitCast>
// CHECK:         `-CStyleCastExpr {{.*}} 'void *__single' <BitCast>
// CHECK:           `-CStyleCastExpr {{.*}} 'char *__single' <BitCast>
// CHECK:             `-ParenExpr {{.*}} 'void *__single'
// CHECK:               `-CStyleCastExpr {{.*}} 'void *__single' <NullToPointer>
// CHECK:                 `-IntegerLiteral {{.*}} 'int' 0
void test_null_to_bidi() {
  int *impl_bidi_ptr = (int *__bidi_indexable)NULL;
  int *impl_bidi_ptr2 = (void *)(char *)NULL;
}
