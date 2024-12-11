

// RUN: %clang_cc1 -fbounds-safety -ast-dump -fbounds-safety-bringup-missing-checks=indirect_count_update %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -ast-dump -fno-bounds-safety-bringup-missing-checks=indirect_count_update %s 2>&1 | FileCheck --check-prefix WITHOUT %s
#include <ptrcheck.h>

// CHECK: |-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: | |-ParmVarDecl [[var_start:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_end:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-BinaryOperator {{.+}} 'int' '='
// CHECK: |   | |-UnaryOperator {{.+}} cannot overflow
// CHECK: |   | | `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |   |-BoundsCheckExpr {{.+}} 'end - 1UL <= __builtin_get_pointer_upper_bound(start + 1UL) && start + 1UL <= end - 1UL && __builtin_get_pointer_lower_bound(start + 1UL) <= start + 1UL'
// CHECK: |   | |   | |-UnaryOperator {{.+}} postfix '--'
// CHECK: |   | |   | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   |   | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   | |       | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | |   |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |     |-GetBoundExpr {{.+}} lower
// CHECK: |   | |   |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |-OpaqueValueExpr [[ove]]
// CHECK: |   | |   | `-DeclRefExpr {{.+}} [[var_end]]
// CHECK: |   | |   |-OpaqueValueExpr [[ove_1]]
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
// CHECK: |   | |   |   |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |   |   | |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |   | |   |   | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK: |   | |   |   | | `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |   | |   |   | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK: |   | |   |   |   `-DeclRefExpr {{.+}} [[var_start]]
// CHECK: |   | |   |   `-IntegerLiteral {{.+}} 1
// CHECK: |   | |   |-OpaqueValueExpr [[ove_3]]
// CHECK: |   | |   | `-DeclRefExpr {{.+}} [[var_start]]
// CHECK: |   | |   `-OpaqueValueExpr [[ove_2]]
// CHECK: |   | |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: |   | |       |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |       | |-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |   | |       | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK: |   | |       | | `-DeclRefExpr {{.+}} [[var_end]]
// CHECK: |   | |       | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK: |   | |       |   `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |   | |       `-IntegerLiteral {{.+}} 1
// CHECK: |   | `-IntegerLiteral {{.+}} 0
// CHECK: |   `-BinaryOperator {{.+}} 'int' '='
// CHECK: |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |   |-UnaryOperator {{.+}} postfix '++'
// CHECK: |     |   | `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |     |   |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |     |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     |   |-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |     |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-IntegerLiteral {{.+}} 0

// WITHOUT: |-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// WITHOUT: | |-ParmVarDecl [[var_start:0x[^ ]+]]
// WITHOUT: | |-ParmVarDecl [[var_end:0x[^ ]+]]
// WITHOUT: | `-CompoundStmt
// WITHOUT: |   |-BinaryOperator {{.+}} 'int' '='
// WITHOUT: |   | |-UnaryOperator {{.+}} cannot overflow
// WITHOUT: |   | | `-UnaryOperator {{.+}} postfix '--'
// WITHOUT: |   | |   `-DeclRefExpr {{.+}} [[var_end]]
// WITHOUT: |   | `-IntegerLiteral {{.+}} 0
// WITHOUT: |   `-BinaryOperator {{.+}} 'int' '='
// WITHOUT: |     |-UnaryOperator {{.+}} cannot overflow
// WITHOUT: |     | `-UnaryOperator {{.+}} postfix '++'
// WITHOUT: |     |   `-DeclRefExpr {{.+}} [[var_start]]
// WITHOUT: |     `-IntegerLiteral {{.+}} 0

void foo(int *__ended_by(end) start, int * end) {
    *end-- = 0;
	*start++ = 0;
}

// CHECK: `-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK:   |-ParmVarDecl [[var_start_1:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_end_1:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-UnaryOperator {{.+}} cannot overflow
// CHECK:     | | `-ParenExpr
// CHECK:     | |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |     |-BoundsCheckExpr {{.+}} 'end - 1 <= __builtin_get_pointer_upper_bound(start + 1) && start + 1 <= end - 1 && __builtin_get_pointer_lower_bound(start + 1) <= start + 1'
// CHECK:     | |     | |-BinaryOperator {{.+}} 'int *__single __ended_by(end)':'int *__single' '='
// CHECK:     | |     | | |-DeclRefExpr {{.+}} [[var_start_1]]
// CHECK:     | |     | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     | |     | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   | | `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |     |-GetBoundExpr {{.+}} lower
// CHECK:     | |     |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |-OpaqueValueExpr [[ove_4]]
// CHECK:     | |     | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:     | |     |   |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | |     |   | |-DeclRefExpr {{.+}} [[var_start_1]]
// CHECK:     | |     |   | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK:     | |     |   | | `-DeclRefExpr {{.+}} [[var_end_1]]
// CHECK:     | |     |   | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK:     | |     |   |   `-DeclRefExpr {{.+}} [[var_start_1]]
// CHECK:     | |     |   `-IntegerLiteral {{.+}} 1
// CHECK:     | |     `-OpaqueValueExpr [[ove_5]]
// CHECK:     | |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
// CHECK:     | |         |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | |         | |-DeclRefExpr {{.+}} [[var_end_1]]
// CHECK:     | |         | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK:     | |         | | `-DeclRefExpr {{.+}} [[var_end_1]]
// CHECK:     | |         | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK:     | |         |   `-DeclRefExpr {{.+}} [[var_start_1]]
// CHECK:     | |         `-IntegerLiteral {{.+}} 1
// CHECK:     | `-IntegerLiteral {{.+}} 0
// CHECK:     `-BinaryOperator {{.+}} 'int' '='
// CHECK:       |-UnaryOperator {{.+}} cannot overflow
// CHECK:       | `-ParenExpr
// CHECK:       |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |     |-BinaryOperator {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' '='
// CHECK:       |     | |-DeclRefExpr {{.+}} [[var_end_1]]
// CHECK:       |     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       |     |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:       `-IntegerLiteral {{.+}} 0

// WITHOUT: `-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// WITHOUT:   |-ParmVarDecl [[var_start_1:0x[^ ]+]]
// WITHOUT:   |-ParmVarDecl [[var_end_1:0x[^ ]+]]
// WITHOUT:   `-CompoundStmt
// WITHOUT:     |-BinaryOperator {{.+}} 'int' '='
// WITHOUT:     | |-UnaryOperator {{.+}} cannot overflow
// WITHOUT:     | | `-ParenExpr
// WITHOUT:     | |   `-BinaryOperator {{.+}} 'int *__single __ended_by(end)':'int *__single' '='
// WITHOUT:     | |     |-DeclRefExpr {{.+}} [[var_start_1]]
// WITHOUT:     | |     `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// WITHOUT:     | |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// WITHOUT:     | |         |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// WITHOUT:     | |         | |-DeclRefExpr {{.+}} [[var_start_1]]
// WITHOUT:     | |         | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// WITHOUT:     | |         | | `-DeclRefExpr {{.+}} [[var_end_1]]
// WITHOUT:     | |         | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// WITHOUT:     | |         |   `-DeclRefExpr {{.+}} [[var_start_1]]
// WITHOUT:     | |         `-IntegerLiteral {{.+}} 1
// WITHOUT:     | `-IntegerLiteral {{.+}} 0
// WITHOUT:     `-BinaryOperator {{.+}} 'int' '='
// WITHOUT:       |-UnaryOperator {{.+}} cannot overflow
// WITHOUT:       | `-ParenExpr
// WITHOUT:       |   `-BinaryOperator {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' '='
// WITHOUT:       |     |-DeclRefExpr {{.+}} [[var_end_1]]
// WITHOUT:       |     `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <BoundsSafetyPointerCast>
// WITHOUT:       |       `-BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
// WITHOUT:       |         |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// WITHOUT:       |         | |-DeclRefExpr {{.+}} [[var_end_1]]
// WITHOUT:       |         | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// WITHOUT:       |         | | `-DeclRefExpr {{.+}} [[var_end_1]]
// WITHOUT:       |         | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// WITHOUT:       |         |   `-DeclRefExpr {{.+}} [[var_start_1]]
// WITHOUT:       |         `-IntegerLiteral {{.+}} 1
// WITHOUT:       `-IntegerLiteral {{.+}} 0

void bar(int *__ended_by(end) start, int * end) {
	*(start = start+1) = 0;
    *(end = end-1) = 0;
}
