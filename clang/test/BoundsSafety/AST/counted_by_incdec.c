

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

struct T {
  int *__counted_by(len) ptr;
  int len;
};

void Test(struct T *t) {
  t->ptr++;
  t->len--;
}

// CHECK-LABEL: Test 'void (struct T *__single)'
// CHECK: | |-ParmVarDecl [[var_t:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr
// CHECK: |   | | |-UnaryOperator {{.+}} postfix '++'
// CHECK: |   | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | | |     `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | | |     | | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   | |   | | |     | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   | |     | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | |     | |     | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | |     | |     |     `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK: |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK: |   | |     |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |     |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   | |       `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove_1]]
// CHECK: |   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK: |   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK: |   | |-OpaqueValueExpr [[ove]]
// CHECK: |   | | `-MemberExpr {{.+}} ->ptr
// CHECK: |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK: |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: |   | | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: |   | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |   | | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   | |   | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |   | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   | |   | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK: |   | |   | | |-OpaqueValueExpr [[ove_4]]
// CHECK: |   | |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |   | | |   `-MemberExpr {{.+}} ->len
// CHECK: |   | |   | | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK: |   | |   | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |   | |   | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |   | |   | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK: |   | |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   | |   `-IntegerLiteral {{.+}} 1
// CHECK: |   | |-OpaqueValueExpr [[ove_7]]
// CHECK: |   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK: |   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK: |   | |-OpaqueValueExpr [[ove_6]]
// CHECK: |   | | `-MemberExpr {{.+}} ->len
// CHECK: |   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct T *__single'
// CHECK: |   | `-OpaqueValueExpr [[ove_5]]
// CHECK: |   |   `-BinaryOperator {{.+}} 'int' '-'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     | `-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK: |   |     `-IntegerLiteral {{.+}} 1
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-UnaryOperator {{.+}} postfix '--'
// CHECK: |     | `-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK: |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK: |     |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'struct T *__single'
// CHECK: |     |-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK: |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'

void TestReverse(struct T *t) {
  t->len--;
  t->ptr++;
}

// CHECK-LABEL: TestReverse 'void (struct T *__single)'
// CHECK: |-ParmVarDecl [[var_t_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-UnaryOperator {{.+}} postfix '--'
// CHECK:   | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | |     `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | |     | | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   | |   | | |     | | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | |   | | |     | | | |       `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | |   | | |     | | | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:   | |     | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK:   | |     |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |       |-IntegerLiteral {{.+}} 0
// CHECK:   | |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:   | |-OpaqueValueExpr [[ove_9]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t_1]]
// CHECK:   | |-OpaqueValueExpr [[ove_8]]
// CHECK:   | | `-MemberExpr {{.+}} ->len
// CHECK:   | |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'struct T *__single'
// CHECK:   | |-OpaqueValueExpr [[ove_15]]
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '-'
// CHECK:   | |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   | |   | `-OpaqueValueExpr [[ove_8]] {{.*}} lvalue
// CHECK:   | |   `-IntegerLiteral {{.+}} 1
// CHECK:   | |-OpaqueValueExpr [[ove_13]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t_1]]
// CHECK:   | |-OpaqueValueExpr [[ove_12]]
// CHECK:   | | `-MemberExpr {{.+}} ->ptr
// CHECK:   | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct T *__single'
// CHECK:   | `-OpaqueValueExpr [[ove_10]]
// CHECK:   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:   |     | | | |-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:   |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   |     | | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   |     | | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:   |     | | |-OpaqueValueExpr [[ove_14]]
// CHECK:   |     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |     | | |   `-MemberExpr {{.+}} ->len
// CHECK:   |     | | |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'struct T *__single'
// CHECK:   |     | | `-OpaqueValueExpr [[ove_11]]
// CHECK:   |     | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK:   |     | |     `-OpaqueValueExpr [[ove_12]] {{.*}} lvalue
// CHECK:   |     | |-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:   |     | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:   |     `-IntegerLiteral {{.+}} 1
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-UnaryOperator {{.+}} postfix '++'
// CHECK:     | `-OpaqueValueExpr [[ove_12]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_9]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove_8]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_13]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove_12]] {{.*}} lvalue
// CHECK:     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
