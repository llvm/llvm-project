

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

struct T {
  int *__ended_by(iter) start;
  int *__ended_by(end) iter;
  int *end;
};

void Test(struct T *t) {
  t->start++;
  --t->iter;
  t->end--;
}

// CHECK-LABEL: Test 'void (struct T *__single)'
// CHECK: |-ParmVarDecl [[var_t:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr {{.+}} 't->end - 1UL <= __builtin_get_pointer_upper_bound(t->iter - 1UL) && t->iter - 1UL <= t->end - 1UL && __builtin_get_pointer_lower_bound(t->iter - 1UL) <= t->iter - 1UL'
// CHECK:   | | |-BoundsCheckExpr {{.+}} 't->iter - 1UL <= __builtin_get_pointer_upper_bound(t->start + 1UL) && t->start + 1UL <= t->iter - 1UL && __builtin_get_pointer_lower_bound(t->start + 1UL) <= t->start + 1UL'
// CHECK:   | | | |-UnaryOperator {{.+}} postfix '++'
// CHECK:   | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | |     `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | | |   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |   | | |     | | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | |   | | |     | | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | | |   | | |     | | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__single /* __started_by(iter) */ ':'int *__single'
// CHECK:   | | |   | | |     | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__single __ended_by(iter)':'int *__single'
// CHECK:   | | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | | |   | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |   | |       | | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | | |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | |     |-GetBoundExpr {{.+}} lower
// CHECK:   | | |     | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | | |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | |     | | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | |   | | |     | | | |   `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'struct T *__single'
// CHECK:   | |   | | |     | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-GetBoundExpr {{.+}} lower
// CHECK:   | |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_1]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK:   | |-OpaqueValueExpr [[ove]]
// CHECK:   | | `-MemberExpr {{.+}} ->start
// CHECK:   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:   | |-OpaqueValueExpr [[ove_7]]
// CHECK:   | | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:   | |   | | | |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK:   | |   | | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   | |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(iter)':'int *__single' <LValueToRValue>
// CHECK:   | |   | | |   `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK:   | |   | | `-OpaqueValueExpr [[ove_8]]
// CHECK:   | |   | |   `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK:   | |   | |     `-MemberExpr {{.+}} ->iter
// CHECK:   | |   | |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:   | |   | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   | |   `-IntegerLiteral {{.+}} 1
// CHECK:   | |-OpaqueValueExpr [[ove_4]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK:   | |-OpaqueValueExpr [[ove_3]]
// CHECK:   | | `-MemberExpr {{.+}} ->iter
// CHECK:   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct T *__single'
// CHECK:   | |-OpaqueValueExpr [[ove_2]]
// CHECK:   | | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
// CHECK:   | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:   | |   | | | |-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK:   | |   | | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single /* __started_by(iter) */ ':'int *__single'
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single __ended_by(iter)':'int *__single'
// CHECK:   | |   | | |-OpaqueValueExpr [[ove_6]]
// CHECK:   | |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(iter)':'int *__single' <LValueToRValue>
// CHECK:   | |   | | |   `-MemberExpr {{.+}} ->start
// CHECK:   | |   | | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct T *__single'
// CHECK:   | |   | | `-OpaqueValueExpr [[ove_5]]
// CHECK:   | |   | |   `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(iter) */ ':'int *__single' <LValueToRValue>
// CHECK:   | |   | |     `-MemberExpr {{.+}} ->end
// CHECK:   | |   | |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'struct T *__single'
// CHECK:   | |   | |-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single __ended_by(iter)':'int *__single'
// CHECK:   | |   | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single /* __started_by(iter) */ ':'int *__single'
// CHECK:   | |   `-IntegerLiteral {{.+}} 1
// CHECK:   | |-OpaqueValueExpr [[ove_11]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'struct T *__single' <LValueToRValue>
// CHECK:   | |   `-DeclRefExpr {{.+}} [[var_t]]
// CHECK:   | |-OpaqueValueExpr [[ove_10]]
// CHECK:   | | `-MemberExpr {{.+}} ->end
// CHECK:   | |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'struct T *__single'
// CHECK:   | `-OpaqueValueExpr [[ove_9]]
// CHECK:   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '-'
// CHECK:   |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:   |     | | | |-OpaqueValueExpr [[ove_10]] {{.*}} lvalue
// CHECK:   |     | | | |-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(iter) */ ':'int *__single' <LValueToRValue>
// CHECK:   |     | | | | `-OpaqueValueExpr [[ove_10]] {{.*}} lvalue
// CHECK:   |     | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   |     | | `-OpaqueValueExpr [[ove_12]]
// CHECK:   |     | |   `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single' <LValueToRValue>
// CHECK:   |     | |     `-MemberExpr {{.+}} ->iter
// CHECK:   |     | |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'struct T *__single'
// CHECK:   |     | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single'
// CHECK:   |     `-IntegerLiteral {{.+}} 1
// CHECK:   |-UnaryOperator {{.+}} prefix '--'
// CHECK:   | `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-UnaryOperator {{.+}} postfix '--'
// CHECK:     | `-OpaqueValueExpr [[ove_10]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK:     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_11]] {{.*}} 'struct T *__single'
// CHECK:     |-OpaqueValueExpr [[ove_10]] {{.*}} lvalue
// CHECK:     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'

