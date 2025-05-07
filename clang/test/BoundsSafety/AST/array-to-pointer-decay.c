

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

int glen;
int gArr[5];

void Test() {
    int *__bidi_indexable ptrGArr = gArr;
    // CHECK: VarDecl {{.+}} ptrGArr 'int *__bidi_indexable' cinit
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[5]' lvalue Var {{.+}} 'gArr' 'int[5]'

    int arrVLA[glen];
    int *__bidi_indexable ptrVLA = arrVLA;
    ptrVLA = arrVLA;

    int arrLocal[7];

    int *__indexable ptrArrayLocal = arrLocal;
    // CHECK: VarDecl {{.+}} ptrArrayLocal 'int *__indexable' cinit
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'
    ptrArrayLocal = arrLocal;
    // CHECK-NEXT: BinaryOperator {{.+}} 'int *__indexable' '='
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int *__indexable' lvalue Var {{.+}} 'ptrArrayLocal' 'int *__indexable'
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'

    int *__single ptrThinLocal = arrLocal;
    // CHECK: VarDecl {{.+}} ptrThinLocal 'int *__single' cinit
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'
    ptrThinLocal = arrLocal;
    // CHECK-NEXT: BinaryOperator {{.+}} 'int *__single' '='
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int *__single' lvalue Var {{.+}} 'ptrThinLocal' 'int *__single'
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
    // CHECK-NEXT: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'

    int *ptrUnspecifiedLocal = arrLocal;
    // CHECK: VarDecl {{.+}} ptrUnspecifiedLocal 'int *__bidi_indexable'
    // CHECK: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'
    ptrUnspecifiedLocal = arrLocal;
    // CHECK-NEXT: BinaryOperator {{.+}} 'int *__bidi_indexable'{{.*}} '='
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int *__bidi_indexable'{{.*}} 'ptrUnspecifiedLocal' 'int *__bidi_indexable'
    // CHECK: ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
    // CHECK-NEXT: DeclRefExpr {{.+}} 'int[7]' lvalue Var {{.+}} 'arrLocal' 'int[7]'

    int *__bidi_indexable ptrFromArraySub = &arrLocal[0];
}

