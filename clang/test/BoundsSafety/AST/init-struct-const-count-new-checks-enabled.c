// FileCheck lines automatically generated using make-ast-dump-check-v2.py
// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=all -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

// expected-no-diagnostics

// =============================================================================
// __counted_by
// =============================================================================

struct cb {
  const int count;
  int* __counted_by(count) ptr;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_cb 'void (struct cb)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct cb'
void consume_cb(struct cb);

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_cb 'void (int, int *__single __counted_by(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cb' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct cb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | | | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | | |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |   |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'c' 'struct cb'
void init_list_cb(int count_param, int*__counted_by(count_param) ptr) {
  struct cb c = {.count = count_param, .ptr = ptr };
  consume_cb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_cb_bidi 'void (int, int *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cb' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct cb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'c' 'struct cb'
void init_list_cb_bidi(int count_param, int* __bidi_indexable ptr) {
  struct cb c = {.count = count_param, .ptr = ptr };
  consume_cb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_cb 'void (int, int *__single __counted_by(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cb' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct cb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |   |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |           `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |             |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |             | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |             | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |             | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             | | `-<<<NULL>>>
// CHECK-NEXT: |   |             | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             |-OpaqueValueExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             |   `-DeclRefExpr {{.+}} 'int *__single __counted_by(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by(count_param)':'int *__single'
// CHECK-NEXT: |   |             `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |                 `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'c' 'struct cb'
void compound_literal_init_cb(int count_param, int*__counted_by(count_param) ptr) {
  struct cb c = (struct cb){.count = count_param, .ptr = ptr };
  consume_cb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_cb_bidi 'void (int, int *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cb' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct cb' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct cb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct cb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |   |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         |   |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cb)' Function {{.+}} 'consume_cb' 'void (struct cb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cb' lvalue Var {{.+}} 'c' 'struct cb'
void compound_literal_init_cb_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cb c = (struct cb){.count = count_param, .ptr = ptr };
  consume_cb(c);
}

// =============================================================================
// __counted_by_or_null
// =============================================================================

struct cbon {
  const int count;
  int* __counted_by_or_null(count) ptr;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_cbon 'void (struct cbon)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct cbon'
void consume_cbon(struct cbon);

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_cbon 'void (int, int *__single __counted_by_or_null(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cbon' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct cbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct cbon'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | | | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | | |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     | |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |     |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |     |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |     |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |     |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |     |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |     |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |     |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |     |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |     |     |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |     |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |     |     |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cbon)' Function {{.+}} 'consume_cbon' 'void (struct cbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cbon' lvalue Var {{.+}} 'c' 'struct cbon'
void init_list_cbon(int count_param, int*__counted_by_or_null(count_param) ptr) {
  struct cbon c = {.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_cbon_bidi 'void (int, int *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cbon' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct cbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct cbon'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |     |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |     |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |     |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |     |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cbon)' Function {{.+}} 'consume_cbon' 'void (struct cbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cbon' lvalue Var {{.+}} 'c' 'struct cbon'
void init_list_cbon_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cbon c = {.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_cbon 'void (int, int *__single __counted_by_or_null(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cbon' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct cbon' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct cbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct cbon'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | | |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | | |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | |       |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |       | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | | |       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | |       | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | |       | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         | |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |         |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |         |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |     |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   |   `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |     |   |     |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |     |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   |     | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |     |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |   |     | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |     |   |     | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   |     |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |       `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |     |         |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |     |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |         | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |         |     |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |         | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |     |         | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |         |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |         |     |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |           `-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |             |-MaterializeSequenceExpr {{.+}} 'int *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |             | | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: |   |             | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |             | | | | `-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | | |     `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             | | `-<<<NULL>>>
// CHECK-NEXT: |   |             | |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             |-OpaqueValueExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' <LValueToRValue>
// CHECK-NEXT: |   |             |   `-DeclRefExpr {{.+}} 'int *__single __counted_by_or_null(count_param)':'int *__single' lvalue ParmVar {{.+}} 'ptr' 'int *__single __counted_by_or_null(count_param)':'int *__single'
// CHECK-NEXT: |   |             `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |                 `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cbon)' Function {{.+}} 'consume_cbon' 'void (struct cbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cbon' lvalue Var {{.+}} 'c' 'struct cbon'
void compound_literal_init_cbon(int count_param, int*__counted_by_or_null(count_param) ptr) {
  struct cbon c = (struct cbon){.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_cbon_bidi 'void (int, int *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'int *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct cbon' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct cbon' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct cbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= __builtin_get_pointer_upper_bound(ptr) - ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct cbon'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'int *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |         |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |         |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |     |   |-GetBoundExpr {{.+}} 'int *' upper
// CHECK-NEXT: |   |         |     |   | `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |     `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |     `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |     |         `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'int *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct cbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct cbon)' Function {{.+}} 'consume_cbon' 'void (struct cbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct cbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct cbon' lvalue Var {{.+}} 'c' 'struct cbon'
void compound_literal_init_cbon_bidi(int count_param, int*__bidi_indexable ptr) {
  struct cbon c = (struct cbon){.count = count_param, .ptr = ptr };
  consume_cbon(c);
}

// =============================================================================
// __sized_by
// =============================================================================

struct sb {
  const int count;
  char* __sized_by(count) ptr;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_sb 'void (struct sb)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct sb'
void consume_sb(struct sb);

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_sb 'void (int, char *__single __sized_by(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sb' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct sb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |     |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |     |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'c' 'struct sb'
void init_list_sb(int count_param, char*__sized_by(count_param) ptr) {
  struct sb c = {.count = count_param, .ptr = ptr };
  consume_sb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_bidi 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sb' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct sb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |     |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |     |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'c' 'struct sb'
void init_list_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sb c = {.count = count_param, .ptr = ptr };
  consume_sb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_sb 'void (int, char *__single __sized_by(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sb' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct sb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |         |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |   |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |         |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |   |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |           | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |           `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |             |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |             | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |             | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             | | `-<<<NULL>>>
// CHECK-NEXT: |   |             | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             |-OpaqueValueExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             |   `-DeclRefExpr {{.+}} 'char *__single __sized_by(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by(count_param)':'char *__single'
// CHECK-NEXT: |   |             `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |                 `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'c' 'struct sb'
void compound_literal_init_sb(int count_param, char*__sized_by(count_param) ptr) {
  struct sb c = (struct sb){.count = count_param, .ptr = ptr };
  consume_sb(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_sb_bidi 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sb' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct sb' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct sb' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct sb'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |   | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |   |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |         |   |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         |   |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |   |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |         |   |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         |   |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |     |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sb)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sb)' Function {{.+}} 'consume_sb' 'void (struct sb)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sb' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sb' lvalue Var {{.+}} 'c' 'struct sb'
void compound_literal_init_sb_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sb c = (struct sb){.count = count_param, .ptr = ptr };
  consume_sb(c);
}

// =============================================================================
// __sized_by_or_null
// =============================================================================

struct sbon {
  const int count;
  char* __sized_by_or_null(count) ptr;
};

// CHECK-LABEL:|-FunctionDecl {{.+}} used consume_sbon 'void (struct sbon)'
// CHECK-NEXT: | `-ParmVarDecl {{.+}} 'struct sbon'
void consume_sbon(struct sbon);

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_sbon 'void (int, char *__single __sized_by_or_null(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sbon' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct sbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct sbon'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |     | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |     |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |     |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |     |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |     |     |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     |     |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |     |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |     |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |     |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |     |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |     |     |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |     |     |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |     |     |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |     |     |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |           | | `-<<<NULL>>>
// CHECK-NEXT: |   |     |     |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |     |     |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sbon)' Function {{.+}} 'consume_sbon' 'void (struct sbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sbon' lvalue Var {{.+}} 'c' 'struct sbon'
void init_list_sbon(int count_param, char*__sized_by_or_null(count_param) ptr) {
  struct sbon c = {.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} init_list_sbon_bidi 'void (int, char *__bidi_indexable)'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__bidi_indexable'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sbon' cinit
// CHECK-NEXT: |   |   `-BoundsCheckExpr {{.+}} 'struct sbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |     |-InitListExpr {{.+}} 'struct sbon'
// CHECK-NEXT: |   |     | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |     | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |     |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |     |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |     |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |     |     |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |     |     |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |     |     |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |     |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |     |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |     |     |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |     |     |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   |     |     `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |     |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |     |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |     |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: |   |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sbon)' Function {{.+}} 'consume_sbon' 'void (struct sbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sbon' lvalue Var {{.+}} 'c' 'struct sbon'
void init_list_sbon_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sbon c = {.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

// CHECK-LABEL:|-FunctionDecl {{.+}} compound_literal_init_sbon 'void (int, char *__single __sized_by_or_null(count_param))'
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT: | | `-DependerDeclsAttr {{.+}} <<invalid sloc>> Implicit {{.+}} 0
// CHECK-NEXT: | |-ParmVarDecl {{.+}} used ptr 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: | `-CompoundStmt {{.+}}
// CHECK-NEXT: |   |-DeclStmt {{.+}}
// CHECK-NEXT: |   | `-VarDecl {{.+}} used c 'struct sbon' cinit
// CHECK-NEXT: |   |   `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT: |   |     `-CompoundLiteralExpr {{.+}} 'struct sbon' lvalue
// CHECK-NEXT: |   |       `-BoundsCheckExpr {{.+}} 'struct sbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT: |   |         |-InitListExpr {{.+}} 'struct sbon'
// CHECK-NEXT: |   |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | | |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | | |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | | |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | | |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | | |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | | |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | | |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | | |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | | |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | | |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | | |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | | |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | | |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | | |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | | |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT: |   |         | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |       `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         | |         |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         | |         | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         | |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         | |         | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         | | `-<<<NULL>>>
// CHECK-NEXT: |   |         | |         | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | |         |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         | |         |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         | |         `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         | |             `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: |   |         |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT: |   |         |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |   `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |   |     |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |   |     | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |   |     | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |     | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |     | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |   |     | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   |     |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |   |     |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |   |     `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |   |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   |         `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: |   |         |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: |   |         |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: |   |         |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: |   |         |     |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT: |   |         |     |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT: |   |         |     |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   |     `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |     |   |       |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |     |   |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     |   |       | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |     |   |       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |   |       | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |       | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   |       | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |     |   |       | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   |       |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |       |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |   |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |   |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT: |   |         |     |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     |         `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |         |     |           |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |         |     |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |         |     |           | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |         |     |           | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |         |     |           | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |           | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |           | | `-<<<NULL>>>
// CHECK-NEXT: |   |         |     |           | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     |           |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |         |     |           |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |         |     |           `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |     |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |     |               `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |     `-BinaryOperator {{.+}} <<invalid sloc>, line:{{.+}}> 'int' '<='
// CHECK-NEXT: |   |         |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: |   |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |         `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |           `-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Unbind>
// CHECK-NEXT: |   |             |-MaterializeSequenceExpr {{.+}} 'char *__bidi_indexable' <Bind>
// CHECK-NEXT: |   |             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: |   |             | | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: |   |             | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: |   |             | | | | `-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | |   `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | | | |     `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             | | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             | | `-<<<NULL>>>
// CHECK-NEXT: |   |             | |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             | |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |             |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |             |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   |             |-OpaqueValueExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' <LValueToRValue>
// CHECK-NEXT: |   |             |   `-DeclRefExpr {{.+}} 'char *__single __sized_by_or_null(count_param)':'char *__single' lvalue ParmVar {{.+}} 'ptr' 'char *__single __sized_by_or_null(count_param)':'char *__single'
// CHECK-NEXT: |   |             `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT: |   |               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: |   |                 `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT: |   `-CallExpr {{.+}} 'void'
// CHECK-NEXT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sbon)' <FunctionToPointerDecay>
// CHECK-NEXT: |     | `-DeclRefExpr {{.+}} 'void (struct sbon)' Function {{.+}} 'consume_sbon' 'void (struct sbon)'
// CHECK-NEXT: |     `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT: |       `-DeclRefExpr {{.+}} 'struct sbon' lvalue Var {{.+}} 'c' 'struct sbon'
void compound_literal_init_sbon(int count_param, char*__sized_by_or_null(count_param) ptr) {
  struct sbon c = (struct sbon){.count = count_param, .ptr = ptr };
  consume_sbon(c);
}

// CHECK-LABEL:`-FunctionDecl {{.+}} compound_literal_init_sbon_bidi 'void (int, char *__bidi_indexable)'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used count_param 'int'
// CHECK-NEXT:   |-ParmVarDecl {{.+}} used ptr 'char *__bidi_indexable'
// CHECK-NEXT:   `-CompoundStmt {{.+}}
// CHECK-NEXT:     |-DeclStmt {{.+}}
// CHECK-NEXT:     | `-VarDecl {{.+}} used c 'struct sbon' cinit
// CHECK-NEXT:     |   `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT:     |     `-CompoundLiteralExpr {{.+}} 'struct sbon' lvalue
// CHECK-NEXT:     |       `-BoundsCheckExpr {{.+}} 'struct sbon' 'ptr <= __builtin_get_pointer_upper_bound(ptr) && __builtin_get_pointer_lower_bound(ptr) <= ptr && !ptr || count_param <= (char *)__builtin_get_pointer_upper_bound(ptr) - (char *__bidi_indexable)ptr && 0 <= count_param'
// CHECK-NEXT:     |         |-InitListExpr {{.+}} 'struct sbon'
// CHECK-NEXT:     |         | |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:     |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:     |         | |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT:     |         | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by_or_null(count)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT:     |         |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:     |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:     |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:     |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:     |         | | | | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         | | | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         | | | |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         | | | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT:     |         | | |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         | | |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         | | |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:     |         | |   |-GetBoundExpr {{.+}} 'char *' lower
// CHECK-NEXT:     |         | |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         | |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         | |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:     |         | |     `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         | |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         | |         `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT:     |         |   |-UnaryOperator {{.+}} 'int' prefix '!' cannot overflow
// CHECK-NEXT:     |         |   | `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         |   |     `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT:     |         |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT:     |         |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT:     |         |     | | `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:     |         |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:     |         |     | |     `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT:     |         |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT:     |         |     |   |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK-NEXT:     |         |     |   | `-GetBoundExpr {{.+}} 'char *' upper
// CHECK-NEXT:     |         |     |   |   `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         |     |   |     `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         |     |   |       `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT:     |         |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK-NEXT:     |         |     |       `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |         |     |         `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |         |     |           `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     |         |     `-BinaryOperator {{.+}} <<invalid sloc>, col:{{.+}}> 'int' '<='
// CHECK-NEXT:     |         |       |-IntegerLiteral {{.+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT:     |         |       `-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:     |         |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:     |         |           `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT:     |         |-OpaqueValueExpr {{.+}} 'int'
// CHECK-NEXT:     |         | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT:     |         |   `-DeclRefExpr {{.+}} 'int' lvalue ParmVar {{.+}} 'count_param' 'int'
// CHECK-NEXT:     |         `-OpaqueValueExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT:     |           `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT:     |             `-DeclRefExpr {{.+}} 'char *__bidi_indexable' lvalue ParmVar {{.+}} 'ptr' 'char *__bidi_indexable'
// CHECK-NEXT:     `-CallExpr {{.+}} 'void'
// CHECK-NEXT:       |-ImplicitCastExpr {{.+}} 'void (*__single)(struct sbon)' <FunctionToPointerDecay>
// CHECK-NEXT:       | `-DeclRefExpr {{.+}} 'void (struct sbon)' Function {{.+}} 'consume_sbon' 'void (struct sbon)'
// CHECK-NEXT:       `-ImplicitCastExpr {{.+}} 'struct sbon' <LValueToRValue>
// CHECK-NEXT:         `-DeclRefExpr {{.+}} 'struct sbon' lvalue Var {{.+}} 'c' 'struct sbon'
void compound_literal_init_sbon_bidi(int count_param, char*__bidi_indexable ptr) {
  struct sbon c = (struct sbon){.count = count_param, .ptr = ptr };
  consume_sbon(c);
}
