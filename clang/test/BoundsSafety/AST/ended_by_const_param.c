// FileCheck lines automatically generated using make-ast-dump-check-v2.py
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-apple-mac -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

struct S {
    int *end;
    int *__ended_by(end) iter;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used foo 'void (int *__single __ended_by(end)const, int *__single /* __started_by(start) */ const)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used start 'int *__single __ended_by(end)const':'int *__singleconst'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used end 'int *__single /* __started_by(start) */ const':'int *__singleconst'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-DeclStmt {{.+}}
// CHECK-NEXT: |     `-VarDecl {{.+}} local 'int *__bidi_indexable' cinit
// CHECK-NEXT: |       `-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |         |-DeclRefExpr {{.+}} 'int *__single /* __started_by(start) */ const':'int *__singleconst' lvalue ParmVar {{.+}} 'end' 'int *__single /* __started_by(start) */ const':'int *__singleconst'
// CHECK-NEXT: |         |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ const':'int *__singleconst' <LValueToRValue>
// CHECK-NEXT: |         | `-DeclRefExpr {{.+}} 'int *__single /* __started_by(start) */ const':'int *__singleconst' lvalue ParmVar {{.+}} 'end' 'int *__single /* __started_by(start) */ const':'int *__singleconst'
// CHECK-NEXT: |         `-ImplicitCastExpr {{.+}} <<invalid sloc>> 'int *__single __ended_by(end)const':'int *__singleconst' <LValueToRValue>
// CHECK-NEXT: |           `-DeclRefExpr {{.+}} <<invalid sloc>> 'int *__single __ended_by(end)const':'int *__singleconst' lvalue ParmVar {{.+}} 'start' 'int *__single __ended_by(end)const':'int *__singleconst'
void foo(int * const __ended_by(end) start, int* const end) {
  int *local = end;
}

// CHECK-LABEL:`-FunctionDecl {{.+}} bar 'void (void)'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     |-DeclStmt {{.+}}
// CHECK-NEXT:     | `-VarDecl {{.+}} used arr 'int[40]'
// CHECK-NEXT:     `-MaterializeSequenceExpr {{.+}} 'void' <Unbind>
// CHECK-NEXT:       |-MaterializeSequenceExpr {{.+}} 'void' <Bind>
// CHECK-NEXT:       | |-BoundsCheckExpr {{.+}} 'void' 'arr + 40 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 40 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT:       | | |-CallExpr {{.+}} 'void'
// CHECK-NEXT:       | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __ended_by(end)const, int *__single /* __started_by(start) */ const)' <FunctionToPointerDecay>
// CHECK-NEXT:       | | | | `-DeclRefExpr {{.+}} 'void (int *__single __ended_by(end)const, int *__single /* __started_by(start) */ const)' Function {{.+}} 'foo' 'void (int *__single __ended_by(end)const, int *__single /* __started_by(start) */ const)'
// CHECK-NEXT:       | | | |-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | | | |     `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | | | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | | |       | `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | | |       `-IntegerLiteral {{.+}} 'int' 40
// CHECK-NEXT:       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |   | | |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | |   | | |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |   | | |     | `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |   | | |     `-IntegerLiteral {{.+}} 'int' 40
// CHECK-NEXT:       | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   | |   `-GetBoundExpr {{.+}} 'int *__bidi_indexable' upper
// CHECK-NEXT:       | |   | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |   | |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |   | |         `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |   |   |     `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |   |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | |   |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |   |         | `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |   |         `-IntegerLiteral {{.+}} 'int' 40
// CHECK-NEXT:       | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | |     |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |     | `-GetBoundExpr {{.+}} 'int *__bidi_indexable' lower
// CHECK-NEXT:       | |     |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |     |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |     |       `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | |       `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | |         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |           `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | |-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       | |   `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       |     | `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       |     `-IntegerLiteral {{.+}} 'int' 40
// CHECK-NEXT:       |-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:       |   `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:       `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:         `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:           |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT:           | `-DeclRefExpr {{.+}} 'int[40]' lvalue Var {{.+}} 'arr' 'int[40]'
// CHECK-NEXT:           `-IntegerLiteral {{.+}} 'int' 40
void bar(void) {
  int arr[40];
  foo(arr, arr + 40);
}
