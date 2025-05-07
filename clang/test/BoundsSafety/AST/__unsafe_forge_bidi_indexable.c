

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

void Test() {
    int *__bidi_indexable ptr = __unsafe_forge_bidi_indexable(int *, 0, sizeof(int));
    // CHECK: VarDecl {{.+}} ptr 'int *__bidi_indexable' cinit
    // CHECK: CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
    // CHECK-NEXT: ForgePtrExpr {{.+}} 'void *__bidi_indexable'
    // CHECK-NEXT: ParenExpr
    // CHECK-NEXT: IntegerLiteral {{.+}} 'int' 0
}
