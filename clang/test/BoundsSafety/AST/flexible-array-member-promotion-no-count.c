

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[];
};

// CHECK-LABEL: flex_no_bounds_promotion
int flex_no_bounds_promotion(struct flexible *__bidi_indexable flex) {
    return flex->elems[2];
}
// CHECK: | |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} ->elems
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__bidi_indexable' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |         `-IntegerLiteral {{.+}} 2

// CHECK-LABEL: flex_promote_from_sized_by
int flex_promote_from_sized_by(struct flexible *__sized_by(size) flex, long size) {
    return flex->elems[2];
}
// CHECK: | |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} ->elems
// CHECK: |         |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: |         |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK: |         |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         |     | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK: |         |     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'long'
// CHECK: |         |     | |-OpaqueValueExpr [[ove]]
// CHECK: |         |     | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK: |         |     | |   `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: |         |     | `-OpaqueValueExpr [[ove_1]]
// CHECK: |         |     |   `-ImplicitCastExpr {{.+}} 'long' <LValueToRValue>
// CHECK: |         |     |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK: |         |     |-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK: |         |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'long'
// CHECK: |         `-IntegerLiteral {{.+}} 2

// CHECK-LABEL: flex_promote_from_single
int flex_promote_from_single(struct flexible *flex) {
    return flex->elems[2];
}
// CHECK: | |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} ->elems
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK: |         `-IntegerLiteral {{.+}} 2

struct nested_flexible {
    int blah;
    struct flexible flex;
};

// CHECK-LABEL: nested_flex_no_bounds_promotion
int nested_flex_no_bounds_promotion(struct nested_flexible *__bidi_indexable nested) {
    return nested->flex.elems[2];
}
// CHECK: | |-ParmVarDecl [[var_nested:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-MemberExpr {{.+}} ->flex
// CHECK: |         |     `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__bidi_indexable' <LValueToRValue>
// CHECK: |         |       `-DeclRefExpr {{.+}} [[var_nested]]
// CHECK: |         `-IntegerLiteral {{.+}} 2

// CHECK-LABEL: nested_flex_promote_from_sized_by
int nested_flex_promote_from_sized_by(struct nested_flexible *__sized_by(size) nested, long size) {
    return nested->flex.elems[2];
}
// CHECK: | |-ParmVarDecl [[var_nested_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-MemberExpr {{.+}} ->flex
// CHECK: |         |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct nested_flexible *__bidi_indexable'
// CHECK: |         |       | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: |         |       | | |-ImplicitCastExpr {{.+}} 'struct nested_flexible *' <BitCast>
// CHECK: |         |       | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         |       | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         |       | | |   | `-ImplicitCastExpr {{.+}} 'struct nested_flexible *' <BoundsSafetyPointerCast>
// CHECK: |         |       | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: |         |       | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'long'
// CHECK: |         |       | |-OpaqueValueExpr [[ove_2]]
// CHECK: |         |       | | `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single' <LValueToRValue>
// CHECK: |         |       | |   `-DeclRefExpr {{.+}} [[var_nested_1]]
// CHECK: |         |       | `-OpaqueValueExpr [[ove_3]]
// CHECK: |         |       |   `-ImplicitCastExpr {{.+}} 'long' <LValueToRValue>
// CHECK: |         |       |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK: |         |       |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct nested_flexible *__single __sized_by(size)':'struct nested_flexible *__single'
// CHECK: |         |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: |         `-IntegerLiteral {{.+}} 2

// CHECK-LABEL: nested_flex_promote_from_single
int nested_flex_promote_from_single(struct nested_flexible *nested) {
    return nested->flex.elems[2];
}
// CHECK: |-ParmVarDecl [[var_nested_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       `-ArraySubscriptExpr
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:       | `-MemberExpr {{.+}} .elems
// CHECK:       |   `-MemberExpr {{.+}} ->flex
// CHECK:       |     `-ImplicitCastExpr {{.+}} 'struct nested_flexible *__single' <LValueToRValue>
// CHECK:       |       `-DeclRefExpr {{.+}} [[var_nested_2]]
// CHECK:       `-IntegerLiteral {{.+}} 2
