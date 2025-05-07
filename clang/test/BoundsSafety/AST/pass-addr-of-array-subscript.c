

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

// CHECK: FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
void bar(void *__sized_by(len) buf, int len);

// CHECK: FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
void foo(int *__counted_by(len) elems, int len, int idx) {
    bar(&elems[idx], sizeof(elems[idx]));
}

// CHECK: |-ParmVarDecl [[var_elems:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: | `-DependerDeclsAttr
// CHECK: |-ParmVarDecl [[var_idx:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr {{.+}} '&elems[idx] <= __builtin_get_pointer_upper_bound(&elems[idx]) && __builtin_get_pointer_lower_bound(&elems[idx]) <= &elems[idx] && sizeof (elems[idx]) <= (char *)__builtin_get_pointer_upper_bound(&elems[idx]) - (char *)&elems[idx] && 0 <= sizeof (elems[idx])'
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(void *__single __sized_by(len), int)' <FunctionToPointerDecay>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(len)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | | |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | | | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | | | |       `-ArraySubscriptExpr
// CHECK:     | | | |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | | | |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | | | |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | | | |         | | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | | | |         | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | | | |         | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | | |         | | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | | | |         | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | | |         | | |-OpaqueValueExpr [[ove_1]]
// CHECK:     | | | |         | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:     | | | |         | | |   `-DeclRefExpr {{.+}} [[var_elems]]
// CHECK:     | | | |         | | `-OpaqueValueExpr [[ove_2]]
// CHECK:     | | | |         | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | | |         | |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:     | | | |         | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | | | |         | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     | | | |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | | |           `-DeclRefExpr {{.+}} [[var_idx]]
// CHECK:     | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | |           | | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | | |           | | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |     | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |   |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |       `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     | |-OpaqueValueExpr [[ove]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | |     `-ArraySubscriptExpr
// CHECK:     | |       |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |       | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | |       | | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | |       | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | |       | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |       | | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | |       | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     | |       | | |-OpaqueValueExpr [[ove_1]]
// CHECK:     | |       | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:     | |       | | |   `-DeclRefExpr {{.+}} [[var_elems]]
// CHECK:     | |       | | `-OpaqueValueExpr [[ove_2]]
// CHECK:     | |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |       | |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:     | |       | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     | |       | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     | |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |         `-DeclRefExpr {{.+}} [[var_idx]]
// CHECK:     | `-OpaqueValueExpr [[ove_3]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK:     |     `-UnaryExprOrTypeTraitExpr
// CHECK:     |       `-ParenExpr
// CHECK:     |         `-ArraySubscriptExpr
// CHECK:     |           |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |           | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |           | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     |           | | | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     |           | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     |           | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     |           | | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     |           | | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:     |           | | |-OpaqueValueExpr [[ove_4]]
// CHECK:     |           | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:     |           | | |   `-DeclRefExpr {{.+}} [[var_elems]]
// CHECK:     |           | | `-OpaqueValueExpr [[ove_5]]
// CHECK:     |           | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |           | |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:     |           | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:     |           | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:     |           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |             `-DeclRefExpr {{.+}} [[var_idx]]
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
