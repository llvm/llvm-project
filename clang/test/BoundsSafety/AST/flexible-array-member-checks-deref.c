

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

// CHECK-LABEL: not_checking_count_single
int not_checking_count_single(struct flexible *__single flex) {
// CHECK: | |-ParmVarDecl [[var_flex:0x[^ ]+]]
    return (*flex).elems[12];
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} .elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | |           | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |-GetBoundExpr {{.+}} upper
// CHECK: |         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |   | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |   | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | |   | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | | |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | | | |   | | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | `-OpaqueValueExpr [[ove_2]]
// CHECK: |         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | `-OpaqueValueExpr [[ove]]
// CHECK: |         | |   `-ParenExpr
// CHECK: |         | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |         | |         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | |         | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | |         | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | |         | | |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |         | |         | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |         | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | |         | | |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |         | |         | `-OpaqueValueExpr [[ove_1]]
// CHECK: |         | |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |         | |         |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |         | |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct flexible *__single'
// CHECK: |         | `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 12


// CHECK-LABEL: not_checking_count_unsafe
int not_checking_count_unsafe(struct flexible *__unsafe_indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
    return (*flex).elems[12];
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} .elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | |-GetBoundExpr {{.+}} upper
// CHECK: |         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |   | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |   | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | |   | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | | |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | | | |   | | |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | `-OpaqueValueExpr [[ove_4]]
// CHECK: |         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | | | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |         | |   `-ParenExpr
// CHECK: |         | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | |       `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK: |         | |         `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: |         | `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: checking_count_indexable
int checking_count_indexable(struct flexible *__indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
    return (*flex).elems[12];
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} .elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | |           | | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | | | |-GetBoundExpr {{.+}} upper
// CHECK: |         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |   | | |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |   | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | |   | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   | | |   |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | | |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | | | |   | | |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | `-OpaqueValueExpr [[ove_7]]
// CHECK: |         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | | | |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | `-OpaqueValueExpr [[ove_5]]
// CHECK: |         | |   `-ParenExpr
// CHECK: |         | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__indexable' <FlexibleArrayCountDeref(BasePtr, FamPtr, Count)>
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | |         | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |         | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | `-OpaqueValueExpr [[ove_6]]
// CHECK: |         | |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         | |         |     `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK: |         | |         `-OpaqueValueExpr [[ove_6]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: checking_count_bidi_indexable
int checking_count_bidi_indexable(struct flexible *__indexable flex) {
// CHECK: | |-ParmVarDecl [[var_flex_3:0x[^ ]+]]
    return (*flex).elems[12];
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} .elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | |           | | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | | | |-GetBoundExpr {{.+}} upper
// CHECK: |         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |   | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |   | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | |   | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   | | |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | | |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | | | |   | | |       `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | `-OpaqueValueExpr [[ove_10]]
// CHECK: |         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | | | |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} lvalue
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | `-OpaqueValueExpr [[ove_8]]
// CHECK: |         | |   `-ParenExpr
// CHECK: |         | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__indexable' <FlexibleArrayCountDeref(BasePtr, FamPtr, Count)>
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | |         | | |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |         | |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | |         | `-OpaqueValueExpr [[ove_9]]
// CHECK: |         | |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         | |         |     `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK: |         | |         `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct flexible *__indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_8]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 12


// CHECK-LABEL: checking_count_sized_by
int checking_count_sized_by(struct flexible *__sized_by(size) flex, int size) {
// CHECK: | |-ParmVarDecl [[var_flex_4:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size:0x[^ ]+]]
    return (*flex).elems[12];
}
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | | `-MemberExpr {{.+}} .elems
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | |           | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |           | | |   | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | | | |           | | |   | | |   `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | | |-GetBoundExpr {{.+}} upper
// CHECK: |         | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | | | |   | | |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |         | | | |   | | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | | | |   | | |   | `-MemberExpr {{.+}} ->elems
// CHECK: |         | | | |   | | |   |   `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | | | |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: |         | | | |   | | |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | | |   | `-OpaqueValueExpr [[ove_15]]
// CHECK: |         | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | | | |   |     `-OpaqueValueExpr [[ove_11]] {{.*}} lvalue
// CHECK: |         | | | |   `-OpaqueValueExpr [[ove_15]] {{.*}} 'struct flexible *__single'
// CHECK: |         | | `-OpaqueValueExpr [[ove_11]]
// CHECK: |         | |   `-ParenExpr
// CHECK: |         | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         | |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |         | |-PredefinedBoundsCheckExpr {{.+}} 'struct flexible *__bidi_indexable' <FlexibleArrayCountDeref(BasePtr, FamPtr, Count)>
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         | | |-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: |         | |         | | | `-MemberExpr {{.+}} ->elems
// CHECK: |         | |         | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |         | |   `-MemberExpr {{.+}} ->count
// CHECK: |         | |         | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         | `-OpaqueValueExpr [[ove_12]]
// CHECK: |         | |         |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |         |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |         |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         | |         |     | | |-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | |         |     | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: |         | |         |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | |         |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         | |         |     | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: |         | |         |     | | |   |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | |         |     | | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: |         | |         |     | |-OpaqueValueExpr [[ove_13]]
// CHECK: |         | |         |     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: |         | |         |     | |   `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK: |         | |         |     | `-OpaqueValueExpr [[ove_14]]
// CHECK: |         | |         |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |         |     |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: |         | |         |     |-OpaqueValueExpr [[ove_13]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         | |         |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: |         | |         `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct flexible *__bidi_indexable'
// CHECK: |         | `-OpaqueValueExpr [[ove_11]] {{.*}} lvalue
// CHECK: |         `-IntegerLiteral {{.+}} 12
;
