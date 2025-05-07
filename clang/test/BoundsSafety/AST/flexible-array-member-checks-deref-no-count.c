

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[];
};

// CHECK-LABEL: not_checking_count_single
int not_checking_count_single(struct flexible *__single flex) {
    return (*flex).elems[12];
}
// CHECK: | |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-ParenExpr
// CHECK: |         |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         |       `-ImplicitCastExpr {{.+}} 'struct flexible *__single' <LValueToRValue>
// CHECK: |         |         `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: not_checking_count_unsafe
int not_checking_count_unsafe(struct flexible *__unsafe_indexable flex) {
    return (*flex).elems[12];
}
// CHECK: | |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-ParenExpr
// CHECK: |         |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         |       `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK: |         |         `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: checking_count_indexable
int checking_count_indexable(struct flexible *__indexable flex) {
    return (*flex).elems[12];
}
// CHECK: | |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-ParenExpr
// CHECK: |         |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         |       `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         |         `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: checking_count_bidi_indexable
int checking_count_bidi_indexable(struct flexible *__indexable flex) {
    return (*flex).elems[12];
}
// CHECK: | |-ParmVarDecl [[var_flex_3:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |         | `-MemberExpr {{.+}} .elems
// CHECK: |         |   `-ParenExpr
// CHECK: |         |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         |       `-ImplicitCastExpr {{.+}} 'struct flexible *__indexable' <LValueToRValue>
// CHECK: |         |         `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK: |         `-IntegerLiteral {{.+}} 12

// CHECK-LABEL: checking_count_sized_by
int checking_count_sized_by(struct flexible *__sized_by(size) flex, int size) {
    return (*flex).elems[12];
}
// CHECK: |-ParmVarDecl [[var_flex_4:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | `-DependerDeclsAttr
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       `-ArraySubscriptExpr
// CHECK:         |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:         | `-MemberExpr {{.+}} .elems
// CHECK:         |   `-ParenExpr
// CHECK:         |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:         |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:         |         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         |         | | |-ImplicitCastExpr {{.+}} 'struct flexible *' <BitCast>
// CHECK:         |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:         |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:         |         | | |   | `-ImplicitCastExpr {{.+}} 'struct flexible *' <BoundsSafetyPointerCast>
// CHECK:         |         | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         |         | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:         |         | |-OpaqueValueExpr [[ove]]
// CHECK:         |         | | `-ImplicitCastExpr {{.+}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single' <LValueToRValue>
// CHECK:         |         | |   `-DeclRefExpr {{.+}} [[var_flex_4]]
// CHECK:         |         | `-OpaqueValueExpr [[ove_1]]
// CHECK:         |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         |         |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK:         |         |-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single __sized_by(size)':'struct flexible *__single'
// CHECK:         |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:         `-IntegerLiteral {{.+}} 12
