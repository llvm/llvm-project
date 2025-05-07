

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

// CHECK-LABEL: elem_access
void elem_access(struct flexible *__unsafe_indexable flex) {
  flex->elems[2] = 0; // array subscription check is currently done directly in clang CodeGen.
}
// CHECK: |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-BinaryOperator {{.+}} 'int' '='
// CHECK:     |-ArraySubscriptExpr
// CHECK:     | |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | | | | | `-MemberExpr {{.+}} ->elems
// CHECK:     | | | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__unsafe_indexable'
// CHECK:     | | | | |-GetBoundExpr {{.+}} upper
// CHECK:     | | | | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:     | | | | |   |-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__unsafe_indexable'
// CHECK:     | | | | |   |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | | | | |   | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | | | | |   | | `-MemberExpr {{.+}} ->elems
// CHECK:     | | | | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__unsafe_indexable'
// CHECK:     | | | | |   | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | | | |   |   `-MemberExpr {{.+}} ->count
// CHECK:     | | | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__unsafe_indexable'
// CHECK:     | | | `-OpaqueValueExpr [[ove]]
// CHECK:     | | |   `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK:     | | |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK:     | | `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__unsafe_indexable'
// CHECK:     | `-IntegerLiteral {{.+}} 2
// CHECK:     `-IntegerLiteral {{.+}} 0


// CHECK-LABEL: count_access
void count_access(struct flexible *__unsafe_indexable flex) {
  flex->count = 10;
}
// CHECK: |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | | |-MemberExpr {{.+}} ->count
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK:     | | |   `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK:     | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     `-OpaqueValueExpr [[ove_2]]
// CHECK:       `-IntegerLiteral {{.+}} 10


// CHECK-LABEL: elem_access_deref
void elem_access_deref(struct flexible *__unsafe_indexable flex) {
  (*flex).elems[2] = 0;
}
// CHECK: |-ParmVarDecl [[var_flex_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-BinaryOperator {{.+}} 'int' '='
// CHECK:     |-ArraySubscriptExpr
// CHECK:     | |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | | | | | `-MemberExpr {{.+}} .elems
// CHECK:     | | | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} lvalue
// CHECK:     | | | | |-GetBoundExpr {{.+}} upper
// CHECK:     | | | | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | | | | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | | | | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK:     | | | | |   | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK:     | | | | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | | | | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | | | | |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK:     | | | | |   | | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:     | | | | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | | | |   | | |   `-MemberExpr {{.+}} ->count
// CHECK:     | | | | |   | | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:     | | | | |   | `-OpaqueValueExpr [[ove_3]]
// CHECK:     | | | | |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | | | | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} lvalue
// CHECK:     | | | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct flexible *__single'
// CHECK:     | | | `-OpaqueValueExpr [[ove_2]]
// CHECK:     | | |   `-ParenExpr
// CHECK:     | | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | | |       `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK:     | | |         `-DeclRefExpr {{.+}} [[var_flex_2]]
// CHECK:     | | `-OpaqueValueExpr [[ove_2]] {{.*}} lvalue
// CHECK:     | `-IntegerLiteral {{.+}} 2
// CHECK:     `-IntegerLiteral {{.+}} 0


// CHECK-LABEL: count_access_deref
void count_access_deref(struct flexible *__unsafe_indexable flex) {
  (*flex).count = 10;
}
// CHECK: |-ParmVarDecl [[var_flex_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BoundsCheckExpr {{.+}} '10 <= (*flex).count && 0 <= 10'
// CHECK:     | | |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | | | |-MemberExpr {{.+}} .count
// CHECK:     | | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} lvalue
// CHECK:     | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | | | |-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:     | | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | | | |   `-MemberExpr {{.+}} .count
// CHECK:     | | | |     `-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK:     | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |   |-IntegerLiteral {{.+}} 0
// CHECK:     | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:     | | `-OpaqueValueExpr [[ove_6]]
// CHECK:     | |   `-ParenExpr
// CHECK:     | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | |       `-ImplicitCastExpr {{.+}} 'struct flexible *__unsafe_indexable' <LValueToRValue>
// CHECK:     | |         `-DeclRefExpr {{.+}} [[var_flex_3]]
// CHECK:     | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:     `-OpaqueValueExpr [[ove_7]]
// CHECK:       `-IntegerLiteral {{.+}} 10
