

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -x objective-c -fexperimental-bounds-safety-objc -ast-dump -fbounds-safety %s | FileCheck %s
#include <ptrcheck.h>

struct T {
    int *__counted_by(len) ptr;
    int len;
};

void Test(struct T *t, int amt) {
    t->ptr += amt;
    t->len -= amt;
}

// CHECK-LABEL: Test 'void (struct T *__single, int)'
// CHECK: |-ParmVarDecl [[var_t:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_amt:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-CompoundAssignOperator {{.+}} ComputeResultTy='int *__single
// CHECK:   | | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | |     | | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   | |   | | |     | | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:   | |     | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |     | |     | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | |     | |     |     `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | |     | |     `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK:   | |     |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |       |-IntegerLiteral {{.+}} 0
// CHECK:   | |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:   | |-OpaqueValueExpr [[ove_2]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_amt]]
// CHECK:   | |-OpaqueValueExpr [[ove_1]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK:   | |-OpaqueValueExpr [[ove]]
// CHECK:   | | `-MemberExpr {{.+}} ->ptr
// CHECK:   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:   | |-OpaqueValueExpr [[ove_3]]
// CHECK:   | | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:   | |   | | | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   | |   | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:   | |   | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   | |   | | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:   | |   | | |-OpaqueValueExpr [[ove_5]]
// CHECK:   | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   | |   | | |   `-MemberExpr {{.+}} ->len
// CHECK:   | |   | | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:   | |   | | `-OpaqueValueExpr [[ove_4]]
// CHECK:   | |   | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:   | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK:   | |   | |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:   | |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:   | |-OpaqueValueExpr [[ove_9]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_amt]]
// CHECK:   | |-OpaqueValueExpr [[ove_8]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK:   | |-OpaqueValueExpr [[ove_7]]
// CHECK:   | | `-MemberExpr {{.+}} ->len
// CHECK:   | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct T *__single'
// CHECK:   | `-OpaqueValueExpr [[ove_6]]
// CHECK:   |   `-BinaryOperator {{.+}} 'int' '-'
// CHECK:   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |     | `-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK:   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-CompoundAssignOperator {{.+}} 'int' '-='
// CHECK:     | |-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK:     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK:     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
