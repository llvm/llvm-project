// FileCheck lines automatically generated using make-ast-dump-check-v2.py

// First make sure the previously buggy diagnostic is present
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -verify -Wno-error=bounds-safety-externally-counted-ptr-arith-constant-count %s
// RUN: %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -verify -Wno-error=bounds-safety-externally-counted-ptr-arith-constant-count %s


// Now make sure the AST is as expected (no errors) and identical with and without the new bounds checks
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -Wno-bounds-safety-externally-counted-ptr-arith-constant-count -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -Wno-bounds-safety-externally-counted-ptr-arith-constant-count -ast-dump %s 2>&1 | FileCheck %s


// Verify we generate the same AST even when the warning is treated as an error, with and without the new bounds checks
// Note: `-verify=default` is needed here, otherwise clang will return a non-zero exit because of the error.
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -verify=default -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -fno-bounds-safety-bringup-missing-checks=all -verify=default -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
// CHECK-LABEL:|-FunctionDecl {{.+}} <{{.+}}:{{.+}}, line:{{.+}}> line:{{.+}} test_cb_unary 'void (int *__single __counted_by(3))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used p 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   `-MaterializeSequenceExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <Bind>
// CHECK-NEXT: |     |-MaterializeSequenceExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <Unbind>
// CHECK-NEXT: |     | |-BoundsCheckExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && 3 <= __builtin_get_pointer_upper_bound(p + 1UL) - p + 1UL && 0 <= 3'
// CHECK-NEXT: |     | | |-UnaryOperator {{.+}} 'int *__single __counted_by(3)':'int *__single' prefix '++'
// CHECK-NEXT: |     | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | | | |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | | | | |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | | | | |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | | | |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | | |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | | | | |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | | |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | | |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | | |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | | |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | | |     | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | | |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | | |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | | |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | | |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | | |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | | |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | | |     `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |     | | | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | | |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | | | |       | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | | | |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | | |       | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | |       | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | | | |       | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | | |       | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | |       | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | |       | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | |       | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | |       | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | |       | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | |       | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | |       | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | | |       | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | | |       | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | | |       |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | | |       `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | | | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |     | | | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | |   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | | |   |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | | |   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | |   |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |   |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | | |   |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | |   |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |   |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |   |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |   |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |   |     | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |   |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |   |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |   |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |   |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |   |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |   |     `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | | |         | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | | |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | | |         | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |         | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | | |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | | |         | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |         | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |         | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |         | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |         | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |         | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |         | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |         | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | | |         | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | | |         | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | | |         |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | | |         `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |     | | |   | |-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT: |     | | |   | | `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |     | | |   | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |     | | |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |     | | |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | |   |   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | |   |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | |   |   |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | |   |   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | |   |   |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |   |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | |   |   |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   |   |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |   |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |   |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |   |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |   |     | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | |   |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |   |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |   |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |   |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |   |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |   |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |   |     `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | |   |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     | | |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     | | |   |         | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     | | |   |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     | | |   |         | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |         | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     | | |   |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     | | |   |         | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |         | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |         | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |         | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |         | | | `-<<<NULL>>>
// CHECK-NEXT: |     | | |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |         | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |         | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     | | |   |         | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | |   |         | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | | |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     | | |   |         |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | |   |         `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     | | |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |     | | |     |-ImplicitCastExpr {{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT: |     | | |     | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |     | | |     `-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT: |     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |     | | |         `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | | `-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT: |     | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |     | |     `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | | `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |     |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |     |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |     |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |     |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |     |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |     |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     |     | | | `-<<<NULL>>>
// CHECK-NEXT: |     |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |     |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |     |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |     |     `-IntegerLiteral {{.+}} 'unsigned long' 1
// CHECK-NEXT: |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |         | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |         | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |         | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |         | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |         | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |         | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |         | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |         | | | `-<<<NULL>>>
// CHECK-NEXT: |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |         | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |         | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |         | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |         | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT: |         | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT: |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |         |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT: |         `-IntegerLiteral {{.+}} 'unsigned long' 1
void test_cb_unary(int* __counted_by(3) p) { // expected-note{{__counted_by attribute is here}}
  // default-note@-1{{__counted_by attribute is here}}
  // default-error@+1{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
  ++p; // expected-warning{{positive pointer arithmetic on '__counted_by' attributed pointer with constant count of 3 always traps}}
}

// CHECK-LABEL:`-FunctionDecl {{.+}} test_cb_binary 'void (int *__single __counted_by(3))'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used p 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     `-MaterializeSequenceExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <Bind>
// CHECK-NEXT:       |-MaterializeSequenceExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <Unbind>
// CHECK-NEXT:       | |-BoundsCheckExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' 'p + 1 <= __builtin_get_pointer_upper_bound(p + 1) && __builtin_get_pointer_lower_bound(p + 1) <= p + 1 && 3 <= __builtin_get_pointer_upper_bound(p + 1) - p + 1 && 0 <= 3'
// CHECK-NEXT:       | | |-CompoundAssignOperator {{.+}} 'int *__single __counted_by(3)':'int *__single' '+=' ComputeLHSTy='int *__single __counted_by(3)':'int *__single' ComputeResultTy='int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | | | |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | | | | |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | | | | |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | | | |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | | |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | | | | |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | | | |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | | |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | | |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | | |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | | |     | | | `-<<<NULL>>>
// CHECK-NEXT:       | | | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | | |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | | |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | | |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | | |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | | |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | | |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | | |       `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT:       | | | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | | |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | | | |       | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | | | |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | | |       | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | |       | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | | | |       | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | | |       | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | |       | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | |       | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | |       | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | |       | | | `-<<<NULL>>>
// CHECK-NEXT:       | | | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | |       | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | |       | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | |       | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | | |       | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | | |       | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | |       |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | | |         `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | | | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT:       | | | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | |   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | | |   |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | | |   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | |   |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |   |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | | |   |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | |   |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |   |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |   |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |   |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |   |     | | | `-<<<NULL>>>
// CHECK-NEXT:       | | | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |   |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |   |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |   |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |   |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |   |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |   |       `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | | |         | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | | |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | | |         | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |         | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | | |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | | |         | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |         | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |         | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |         | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |         | | | `-<<<NULL>>>
// CHECK-NEXT:       | | | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |         | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |         | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |         | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | | |         | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | | |         | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |         |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | | |           `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:       | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:       | | |   | |-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT:       | | |   | | `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT:       | | |   | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT:       | | |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT:       | | |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | |   |   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | |   |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | |   |   |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | |   |   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | |   |   |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |   |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | |   |   |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |   |   |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |   |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |   |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |   |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |   |     | | | `-<<<NULL>>>
// CHECK-NEXT:       | | |   |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |   |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |   |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |   |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |   |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |   |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |   |       `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | |   |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       | | |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       | | |   |         | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       | | |   |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       | | |   |         | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |         | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       | | |   |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       | | |   |         | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |         | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |         | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |         | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |         | | | `-<<<NULL>>>
// CHECK-NEXT:       | | |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |         | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |         | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       | | |   |         | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | |   |         | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | | |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |         |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | |   |           `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | | |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT:       | | |     |-ImplicitCastExpr {{.+}} <<invalid sloc>> 'long' <IntegralCast>
// CHECK-NEXT:       | | |     | `-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT:       | | |     `-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT:       | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT:       | | |         `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | | `-OpaqueValueExpr {{.+}} 'long'
// CHECK-NEXT:       | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT:       | |     `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | | `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | | `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:       |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:       |     | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:       |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:       |     | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       |     | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:       |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:       |     | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       |     | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       |     | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |     | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       |     | | | `-<<<NULL>>>
// CHECK-NEXT:       |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       |     | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       |     | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |     | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:       |     | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       |     | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |     |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:       |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       |       `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:       | `-IntegerLiteral {{.+}} 'int' 1
// CHECK-NEXT:       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:       `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:         `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT:           |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT:           | |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT:           | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT:           | | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:           | | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:           | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT:           | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT:           | | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:           | | | | |     `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:           | | | | |       `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:           | | | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:           | | | `-<<<NULL>>>
// CHECK-NEXT:           | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:           | | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:           | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:           | |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:           | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' <LValueToRValue>
// CHECK-NEXT:           | |   `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue
// CHECK-NEXT:           | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(3)':'int *__single' lvalue ParmVar {{.+}} 'p' 'int *__single __counted_by(3)':'int *__single'
// CHECK-NEXT:           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:           |   `-IntegerLiteral {{.+}} 'int' 3
// CHECK-NEXT:           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:             `-IntegerLiteral {{.+}} 'int' 1
void test_cb_binary(int* __counted_by(3) p) { // expected-note{{__counted_by attribute is here}}
  // default-note@-1{{__counted_by attribute is here}}
  // default-error@+1{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
  p += 1; // expected-warning{{compound addition-assignment on '__counted_by' attributed pointer with constant count of 3 always traps}}
}

