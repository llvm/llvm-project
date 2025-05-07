

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

void TestCountedPtr(int *__counted_by(*len) *ptr, unsigned *len) {
  *ptr += 4;
  *len = *len - 4;
}

// CHECK-LABEL: TestCountedPtr
// CHECK: | |-ParmVarDecl [[var_ptr:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr
// CHECK: |   | | |-CompoundAssignOperator {{.+}} ComputeResultTy='int *__single
// CHECK: |   | | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | | |     | | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK: |   | |   | | |     | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: |   | |     `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | |       |-GetBoundExpr {{.+}} upper
// CHECK: |   | |       | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |       `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |         `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |-OpaqueValueExpr [[ove_1]]
// CHECK: |   | | `-IntegerLiteral {{.+}} 4
// CHECK: |   | |-OpaqueValueExpr [[ove]]
// CHECK: |   | | `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)*__single' <LValueToRValue>
// CHECK: |   | |     `-DeclRefExpr {{.+}} [[var_ptr]]
// CHECK: |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: |   | | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: |   | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |   | | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK: |   | |   | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |   | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK: |   | |   | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned int'
// CHECK: |   | |   | | |-OpaqueValueExpr [[ove_3]]
// CHECK: |   | |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)':'int *__single' <LValueToRValue>
// CHECK: |   | |   | | |   `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |   | |   | | `-OpaqueValueExpr [[ove_4]]
// CHECK: |   | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |   | |   | |     `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   | |   | |       `-ImplicitCastExpr {{.+}} 'unsigned int *__single' <LValueToRValue>
// CHECK: |   | |   | |         `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |   | |   | |-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK: |   | |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned int'
// CHECK: |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |   | `-OpaqueValueExpr [[ove_5]]
// CHECK: |   |   `-BinaryOperator {{.+}} 'unsigned int' '-'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |   |     | `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   |     |   `-ImplicitCastExpr {{.+}} 'unsigned int *__single' <LValueToRValue>
// CHECK: |   |     |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'unsigned int' <IntegralCast>
// CHECK: |   |       `-IntegerLiteral {{.+}} 4
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-BinaryOperator {{.+}} 'unsigned int' '='
// CHECK: |     | |-UnaryOperator {{.+}} cannot overflow
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'unsigned int *__single' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |     | `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned int'
// CHECK: |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |     |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned int'

void TestCount(int *__counted_by(len) ptr, unsigned len) {
  int len2 = len;
  int *__counted_by(len2) ptr2 = ptr;
  
  ptr2 = ptr2;
  len2 /= 4;
}
// CHECK-LABEL: TestCount
// CHECK: | |-ParmVarDecl [[var_ptr_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_len2:0x[^ ]+]]
// CHECK: |   |   |-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |   |   | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |   |   |   `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: |   |   `-DependerDeclsAttr
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_ptr2:0x[^ ]+]]
// CHECK: |   |   `-BoundsCheckExpr
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |     | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |     | | |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |     |     | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     | | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | | | `-GetBoundExpr {{.+}} upper
// CHECK: |   |     | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | |   |-GetBoundExpr {{.+}} lower
// CHECK: |   |     | |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |   | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'long'
// CHECK: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |     |   |   |-GetBoundExpr {{.+}} upper
// CHECK: |   |     |   |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     |   |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |     |     | `-IntegerLiteral {{.+}} 0
// CHECK: |   |     |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'long'
// CHECK: |   |     |-OpaqueValueExpr [[ove_6]]
// CHECK: |   |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |     |   | | |-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |     |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |     |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |     |   | | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |     |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'unsigned int'
// CHECK: |   |     |   | |-OpaqueValueExpr [[ove_7]]
// CHECK: |   |     |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   |     |   | |   `-DeclRefExpr {{.+}} [[var_ptr_1]]
// CHECK: |   |     |   | `-OpaqueValueExpr [[ove_8]]
// CHECK: |   |     |   |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: |   |     |   |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: |   |     |   |-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |     |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'unsigned int'
// CHECK: |   |     `-OpaqueValueExpr [[ove_9]]
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |           `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr
// CHECK: |   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len2)':'int *__single' '='
// CHECK: |   | | | |-DeclRefExpr {{.+}} [[var_ptr2]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | |       | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len2)':'int *__single'
// CHECK: |   | | |       | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   | |   |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   | |     | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | |     | |     | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   | |     | |     `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK: |   | |     |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   | |       `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove_10]]
// CHECK: |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   | |   | | |-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len2)':'int *__single'
// CHECK: |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len2)':'int *__single'
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK: |   | |   | |-OpaqueValueExpr [[ove_11]]
// CHECK: |   | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len2)':'int *__single' <LValueToRValue>
// CHECK: |   | |   | |   `-DeclRefExpr {{.+}} [[var_ptr2]]
// CHECK: |   | |   | `-OpaqueValueExpr [[ove_12]]
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   | |   |     `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK: |   | |   |-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len2)':'int *__single'
// CHECK: |   | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove_15]]
// CHECK: |   | | `-IntegerLiteral {{.+}} 4
// CHECK: |   | |-OpaqueValueExpr [[ove_14]]
// CHECK: |   | | `-DeclRefExpr {{.+}} [[var_len2]]
// CHECK: |   | `-OpaqueValueExpr [[ove_13]]
// CHECK: |   |   `-BinaryOperator {{.+}} 'int' '/'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     | `-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |   |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-CompoundAssignOperator {{.+}} 'int' '/='
// CHECK: |     | |-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |     | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: |     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     |-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: |     |-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'

struct EndedBy {
  int *__ended_by(end) start;
  char *end;
};
void TestRange(struct EndedBy *e) {
  e->end -= 4;
  e->start = e->start;
}
// CHECK-LABEL: TestRange
// CHECK:   |-ParmVarDecl [[var_e:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CompoundAssignOperator {{.+}} */ ':'
// CHECK:     | | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} lvalue
// CHECK:     | | | |   `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'struct EndedBy *__single'
// CHECK:     | | | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |   | |     | | | `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int *__single __ended_by(end)':'int *__single'
// CHECK:     | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     | |   |     | | |-OpaqueValueExpr [[ove_16]] {{.*}} lvalue
// CHECK:     | |   |     | | |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <LValueToRValue>
// CHECK:     | |   |     | | | `-OpaqueValueExpr [[ove_16]] {{.*}} lvalue
// CHECK:     | |   |     | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int *__single __ended_by(end)':'int *__single'
// CHECK:     | |   |     | `-OpaqueValueExpr [[ove_21]]
// CHECK:     | |   |     |   `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK:     | |   |     |     `-MemberExpr {{.+}} ->start
// CHECK:     | |   |     |       `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct EndedBy *__single'
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__single __ended_by(end)':'int *__single'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |     | | |-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'char *__single /* __started_by(start) */ ':'char *__single'
// CHECK:     | |     |     | | |     `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'struct EndedBy *__single'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK:     | |       `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |         `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:     | |-OpaqueValueExpr [[ove_18]]
// CHECK:     | | `-IntegerLiteral {{.+}} 4
// CHECK:     | |-OpaqueValueExpr [[ove_17]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct EndedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_e]]
// CHECK:     | |-OpaqueValueExpr [[ove_16]]
// CHECK:     | | `-MemberExpr {{.+}} ->end
// CHECK:     | |   `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct EndedBy *__single'
// CHECK:     | |-OpaqueValueExpr [[ove_19]]
// CHECK:     | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '-'
// CHECK:     | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     | |   | | | |-OpaqueValueExpr [[ove_16]] {{.*}} lvalue
// CHECK:     | |   | | | |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <LValueToRValue>
// CHECK:     | |   | | | | `-OpaqueValueExpr [[ove_16]] {{.*}} lvalue
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int *__single __ended_by(end)':'int *__single'
// CHECK:     | |   | | `-OpaqueValueExpr [[ove_20]]
// CHECK:     | |   | |   `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK:     | |   | |     `-MemberExpr {{.+}} ->start
// CHECK:     | |   | |       `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct EndedBy *__single'
// CHECK:     | |   | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int *__single __ended_by(end)':'int *__single'
// CHECK:     | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK:     | `-OpaqueValueExpr [[ove_22]]
// CHECK:     |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     |     | | |-MemberExpr {{.+}} ->start
// CHECK:     |     | | | `-ImplicitCastExpr {{.+}} 'struct EndedBy *__single' <LValueToRValue>
// CHECK:     |     | | |   `-DeclRefExpr {{.+}} [[var_e]]
// CHECK:     |     | | |-OpaqueValueExpr [[ove_23]] {{.*}} 'char *__single /* __started_by(start) */ ':'char *__single'
// CHECK:     |     | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <LValueToRValue>
// CHECK:     |     | |   `-MemberExpr {{.+}} ->start
// CHECK:     |     | |     `-ImplicitCastExpr {{.+}} 'struct EndedBy *__single' <LValueToRValue>
// CHECK:     |     | |       `-DeclRefExpr {{.+}} [[var_e]]
// CHECK:     |     | |-OpaqueValueExpr [[ove_24]]
// CHECK:     |     | | `-ImplicitCastExpr {{.+}} 'struct EndedBy *__single' <LValueToRValue>
// CHECK:     |     | |   `-DeclRefExpr {{.+}} [[var_e]]
// CHECK:     |     | `-OpaqueValueExpr [[ove_23]]
// CHECK:     |     |   `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(start) */ ':'char *__single' <LValueToRValue>
// CHECK:     |     |     `-MemberExpr {{.+}} ->end
// CHECK:     |     |       `-OpaqueValueExpr [[ove_24]] {{.*}} 'struct EndedBy *__single'
// CHECK:     |     |-OpaqueValueExpr [[ove_24]] {{.*}} 'struct EndedBy *__single'
// CHECK:     |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'char *__single /* __started_by(start) */ ':'char *__single'
// CHECK:     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |-BinaryOperator {{.+}} 'int *__single __ended_by(end)':'int *__single' '='
// CHECK:       | |-MemberExpr {{.+}} ->start
// CHECK:       | | `-ImplicitCastExpr {{.+}} 'struct EndedBy *__single' <LValueToRValue>
// CHECK:       | |   `-DeclRefExpr {{.+}} [[var_e]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove_17]] {{.*}} 'struct EndedBy *__single'
// CHECK:       |-OpaqueValueExpr [[ove_16]] {{.*}} lvalue
// CHECK:       |-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:       `-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__bidi_indexable'
