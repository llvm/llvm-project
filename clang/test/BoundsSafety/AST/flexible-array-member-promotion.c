

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

int flex_no_bounds_promotion(struct flexible *__bidi_indexable flex) {
    return flex->elems[2];
}
// CHECK-LABEL: flex_no_bounds_promotion
// CHECK: {{^}}| |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}|         | |   `-ImplicitCastExpr {{.+}} 'struct flexible *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|         | |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2

int flex_promote_from_sized_by(struct flexible *__sized_by(size) flex, long size) {
    return flex->elems[2];
}
// CHECK-LABEL: flex_promote_from_sized_by
// CHECK: {{^}}| |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |       | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: {{^}}|         | | | |       | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'long'
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}|         | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | |     | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: {{^}}|         | |     | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: {{^}}|         | |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}|         | |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}|         | |     | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|         | |     | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: {{^}}|         | |     | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: {{^}}|         | |     | |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}|         | |     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: {{^}}|         | |     | |   `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: {{^}}|         | |     | `-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}|         | |     |   `-ImplicitCastExpr {{.+}} 'long' <LValueToRValue>
// CHECK: {{^}}|         | |     |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: {{^}}|         | |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: {{^}}|         | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2

int flex_promote_from_single(struct flexible *flex) {
    return flex->elems[2];
}
// CHECK-LABEL: flex_promote_from_single
// CHECK: {{^}}| |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|         | | | |   | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | |   | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|         | | | |   |   `-MemberExpr {{.+}} ->count
// CHECK: {{^}}|         | | | |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}|         | |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: {{^}}|         | |     `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2

struct nested_flexible {
    int blah;
    struct flexible flex;
};

int nested_flex_no_bounds_promotion(struct nested_flexible *__bidi_indexable nested) {
    return nested->flex.elems[2];
}
// CHECK-LABEL: nested_flex_no_bounds_promotion
// CHECK: {{^}}| |-ParmVarDecl [[var_nested:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} .elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |   | | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|         | | | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   | | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|         | | | |   | | |   `-MemberExpr {{.+}} ->count
// CHECK: {{^}}|         | | | |   | | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | `-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}|         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|         | | | |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}|         | |   `-MemberExpr {{.+}} ->flex
// CHECK: {{^}}|         | |     `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|         | |       `-DeclRefExpr {{.+}} [[var_nested]]
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2

int nested_flex_promote_from_sized_by(struct nested_flexible *__sized_by(size) nested, long size) {
    return nested->flex.elems[2];
}
// CHECK-LABEL: nested_flex_promote_from_sized_by
// CHECK: {{^}}| |-ParmVarDecl [[var_nested_1:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} .elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |         | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: {{^}}|         | | | |         | | |   `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'long'
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |   | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|         | | | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   | | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|         | | | |   | | |   `-MemberExpr {{.+}} ->count
// CHECK: {{^}}|         | | | |   | | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | `-OpaqueValueExpr [[ove_10]]
// CHECK: {{^}}|         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|         | | | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}|         | |   `-MemberExpr {{.+}} ->flex
// CHECK: {{^}}|         | |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct nested_flexible *__bidi_indexable'
// CHECK: {{^}}|         | |       | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: {{^}}|         | |       | | |-ImplicitCastExpr {{.+}} 'struct nested_flexible *' <BitCast>
// CHECK: {{^}}|         | |       | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}|         | |       | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}|         | |       | | |   | `-ImplicitCastExpr {{.+}} 'struct nested_flexible *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|         | |       | | |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: {{^}}|         | |       | | |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'long'
// CHECK: {{^}}|         | |       | |-OpaqueValueExpr [[ove_8]]
// CHECK: {{^}}|         | |       | | `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single' <LValueToRValue>
// CHECK: {{^}}|         | |       | |   `-DeclRefExpr {{.+}} [[var_nested_1]]
// CHECK: {{^}}|         | |       | `-OpaqueValueExpr [[ove_9]]
// CHECK: {{^}}|         | |       |   `-ImplicitCastExpr {{.+}} 'long' <LValueToRValue>
// CHECK: {{^}}|         | |       |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK: {{^}}|         | |       |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: {{^}}|         | |       `-OpaqueValueExpr [[ove_9]] {{.*}} 'long'
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2


int nested_flex_promote_from_single(struct nested_flexible *nested) {
    return nested->flex.elems[2];
}
// CHECK-LABEL: nested_flex_promote_from_single
// CHECK: {{^}}| |-ParmVarDecl [[var_nested_2:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-ReturnStmt
// CHECK: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|       `-ArraySubscriptExpr
// CHECK: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | | `-MemberExpr {{.+}} .elems
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}|         | | | |   | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|         | | | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}|         | | | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}|         | | | |   | | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|         | | | |   | | |   `-MemberExpr {{.+}} ->count
// CHECK: {{^}}|         | | | |   | | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | | |   | `-OpaqueValueExpr [[ove_12]]
// CHECK: {{^}}|         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|         | | | |   |     `-OpaqueValueExpr [[ove_11]] {{.*}} lvalue
// CHECK: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}|         | | `-OpaqueValueExpr [[ove_11]]
// CHECK: {{^}}|         | |   `-MemberExpr {{.+}} ->flex
// CHECK: {{^}}|         | |     `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__single' <LValueToRValue>
// CHECK: {{^}}|         | |       `-DeclRefExpr {{.+}} [[var_nested_2]]
// CHECK: {{^}}|         | `-OpaqueValueExpr [[ove_11]] {{.*}} lvalue
// CHECK: {{^}}|         `-IntegerLiteral {{.+}} 2

;
