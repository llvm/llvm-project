

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

// CHECK-LABEL: preincdec_init
void preincdec_init(char *__sized_by(len) p, unsigned long long len) {
  char *lp = (--len, ++p);
}
// CHECK: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_lp:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |         `-ParenExpr
// CHECK: |           `-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' ','
// CHECK: |             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |             | |-BoundsCheckExpr {{.+}} 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && len - 1ULL <= (char *)__builtin_get_pointer_upper_bound(p + 1UL) - (char *__bidi_indexable)p + 1UL'
// CHECK: |             | | |-UnaryOperator {{.+}} prefix '--'
// CHECK: |             | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |             | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |             | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |             | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   | | |     | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             | |   | | |     | | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} lvalue
// CHECK: |             | |   | | |     | | | |   |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |             | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |             | |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |             | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |     |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |             | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |             | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |             | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |             | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |             | |         |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |             | |             `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |-OpaqueValueExpr [[ove]]
// CHECK: |             | | `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |             | |-OpaqueValueExpr [[ove_5]]
// CHECK: |             | | `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK: |             | |   |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |             | |   | `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |             | |   `-IntegerLiteral {{.+}} 1
// CHECK: |             | |-OpaqueValueExpr [[ove_3]]
// CHECK: |             | | `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |             | `-OpaqueValueExpr [[ove_1]]
// CHECK: |             |   `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |             |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |             |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |             |     | | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |             |     | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             |     | | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | | | | `-AssumptionExpr
// CHECK: |             |     | | | |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long long'
// CHECK: |             |     | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |             |     | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |             |     | | | |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long long'
// CHECK: |             |     | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |             |     | | | |       `-IntegerLiteral {{.+}} 0
// CHECK: |             |     | | |-OpaqueValueExpr [[ove_2]]
// CHECK: |             |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK: |             |     | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |             |     | | `-OpaqueValueExpr [[ove_4]]
// CHECK: |             |     | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |             |     | |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |             |     | |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long long'
// CHECK: |             |     `-IntegerLiteral {{.+}} 1
// CHECK: |             `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |               |-UnaryOperator {{.+}} prefix '++'
// CHECK: |               | `-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |               |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |               |-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned long long'
// CHECK: |               |-OpaqueValueExpr [[ove_3]] {{.*}} lvalue
// CHECK: |               `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'

// CHECK-LABEL: postincdec
void postincdec(char *__sized_by(len) p, unsigned long long len) {
  p++, len--;
}
// CHECK: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK: |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     | |-BoundsCheckExpr {{.+}} 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && len - 1ULL <= (char *)__builtin_get_pointer_upper_bound(p + 1UL) - (char *__bidi_indexable)p + 1UL'
// CHECK: |     | | |-UnaryOperator {{.+}} postfix '++'
// CHECK: |     | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |     | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |   | | |     | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |     | |   | | |     | | | |   |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |     | |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |     | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |     | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |     | |     |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |     | |     |   | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} lvalue
// CHECK: |     | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |     | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |     | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |     | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |     | |         |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |     | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |     | |             `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |     | |-OpaqueValueExpr [[ove]]
// CHECK: |     | | `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |     | |-OpaqueValueExpr [[ove_1]]
// CHECK: |     | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |     | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |     | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |     | |   | | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |     | |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |     | |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |     | |   | | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |     | |   | | | | `-AssumptionExpr
// CHECK: |     | |   | | | |   |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long long'
// CHECK: |     | |   | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |     | |   | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |     | |   | | | |     | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long long'
// CHECK: |     | |   | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |     | |   | | | |       `-IntegerLiteral {{.+}} 0
// CHECK: |     | |   | | |-OpaqueValueExpr [[ove_2]]
// CHECK: |     | |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK: |     | |   | | |   `-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |     | |   | | `-OpaqueValueExpr [[ove_3]]
// CHECK: |     | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |     | |   | |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |     | |   | |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |     | |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long long'
// CHECK: |     | |   `-IntegerLiteral {{.+}} 1
// CHECK: |     | |-OpaqueValueExpr [[ove_5]]
// CHECK: |     | | `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |     | `-OpaqueValueExpr [[ove_4]]
// CHECK: |     |   `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK: |     |     |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |     |     | `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: |     |     `-IntegerLiteral {{.+}} 1
// CHECK: |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       |-UnaryOperator {{.+}} postfix '--'
// CHECK: |       | `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: |       |-OpaqueValueExpr [[ove]] {{.*}} lvalue
// CHECK: |       |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       |-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long long'

// CHECK-LABEL: postincdec_init
void postincdec_init(char *__sized_by(len) p, unsigned long long len) {
  char *lp = (len--, p++);
}
// CHECK: | |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_lp:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |         `-ParenExpr
// CHECK: |           `-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' ','
// CHECK: |             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |             | |-BoundsCheckExpr {{.+}} 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && len - 1ULL <= (char *)__builtin_get_pointer_upper_bound(p + 1UL) - (char *__bidi_indexable)p + 1UL'
// CHECK: |             | | |-UnaryOperator {{.+}} postfix '--'
// CHECK: |             | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} lvalue
// CHECK: |             | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |             | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |             | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |   | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   | | |     | | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             | |   | | |     | | | |   `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} lvalue
// CHECK: |             | |   | | |     | | | |   |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |             | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |             | |   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |             | |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |             | |     |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |             | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |             | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |             | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |             | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |             | |         |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |             | |             `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             | |-OpaqueValueExpr [[ove_6]]
// CHECK: |             | | `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: |             | |-OpaqueValueExpr [[ove_11]]
// CHECK: |             | | `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK: |             | |   |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |             | |   | `-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK: |             | |   `-IntegerLiteral {{.+}} 1
// CHECK: |             | |-OpaqueValueExpr [[ove_9]]
// CHECK: |             | | `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK: |             | `-OpaqueValueExpr [[ove_7]]
// CHECK: |             |   `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |             |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |             |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |             |     | | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |             |     | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |             |     | | | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | | | | `-AssumptionExpr
// CHECK: |             |     | | | |   |-OpaqueValueExpr [[ove_10]] {{.*}} 'unsigned long long'
// CHECK: |             |     | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |             |     | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |             |     | | | |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'unsigned long long'
// CHECK: |             |     | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |             |     | | | |       `-IntegerLiteral {{.+}} 0
// CHECK: |             |     | | |-OpaqueValueExpr [[ove_8]]
// CHECK: |             |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK: |             |     | | |   `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |             |     | | `-OpaqueValueExpr [[ove_10]]
// CHECK: |             |     | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |             |     | |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: |             |     | |-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'unsigned long long'
// CHECK: |             |     `-IntegerLiteral {{.+}} 1
// CHECK: |             `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |               |-UnaryOperator {{.+}} postfix '++'
// CHECK: |               | `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |               |-OpaqueValueExpr [[ove_6]] {{.*}} lvalue
// CHECK: |               |-OpaqueValueExpr [[ove_11]] {{.*}} 'unsigned long long'
// CHECK: |               |-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |               `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'

// CHECK-LABEL: compound_assign_init
void compound_assign_init(char *__sized_by(len) p, unsigned long long len) {
  unsigned long long l = (p+=1, len-=1);
}
// CHECK: | |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_l:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK: |           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |           | |-BoundsCheckExpr {{.+}} 'p + 1 <= __builtin_get_pointer_upper_bound(p + 1) && __builtin_get_pointer_lower_bound(p + 1) <= p + 1 && len - 1 <= (char *)__builtin_get_pointer_upper_bound(p + 1) - (char *__bidi_indexable)p + 1'
// CHECK: |           | | |-CompoundAssignOperator {{.+}} ComputeResultTy='char *__single
// CHECK: |           | | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} lvalue
// CHECK: |           | | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK: |           | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |           | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |           | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |   | | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   | | |     | | | |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |           | |   | | |     | | | |   |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |           | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |           | |   | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |           | |   |   | `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |   |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |     |-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |           | |     |   | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} lvalue
// CHECK: |           | |     |   `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |           | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |           | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |           | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |           | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |           | |         |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |           | |             `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |-OpaqueValueExpr [[ove_13]]
// CHECK: |           | | `-IntegerLiteral {{.+}} 1
// CHECK: |           | |-OpaqueValueExpr [[ove_12]]
// CHECK: |           | | `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK: |           | |-OpaqueValueExpr [[ove_14]]
// CHECK: |           | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |           | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |           | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |           | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |           | |   | | | |-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |           | |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |           | |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |   | | | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |           | |   | | | | `-AssumptionExpr
// CHECK: |           | |   | | | |   |-OpaqueValueExpr [[ove_16]] {{.*}} 'unsigned long long'
// CHECK: |           | |   | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |           | |   | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |           | |   | | | |     | `-OpaqueValueExpr [[ove_16]] {{.*}} 'unsigned long long'
// CHECK: |           | |   | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |           | |   | | | |       `-IntegerLiteral {{.+}} 0
// CHECK: |           | |   | | |-OpaqueValueExpr [[ove_15]]
// CHECK: |           | |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK: |           | |   | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} lvalue
// CHECK: |           | |   | | `-OpaqueValueExpr [[ove_16]]
// CHECK: |           | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |           | |   | |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: |           | |   | |-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |           | |   | `-OpaqueValueExpr [[ove_16]] {{.*}} 'unsigned long long'
// CHECK: |           | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK: |           | |-OpaqueValueExpr [[ove_19]]
// CHECK: |           | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |           | |   `-IntegerLiteral {{.+}} 1
// CHECK: |           | |-OpaqueValueExpr [[ove_18]]
// CHECK: |           | | `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: |           | `-OpaqueValueExpr [[ove_17]]
// CHECK: |           |   `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK: |           |     |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |           |     | `-OpaqueValueExpr [[ove_18]] {{.*}} lvalue
// CHECK: |           |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'unsigned long long'
// CHECK: |           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             |-CompoundAssignOperator {{.+}} long' ComputeResultTy='unsigned
// CHECK: |             | |-OpaqueValueExpr [[ove_18]] {{.*}} lvalue
// CHECK: |             | `-OpaqueValueExpr [[ove_19]] {{.*}} 'unsigned long long'
// CHECK: |             |-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK: |             |-OpaqueValueExpr [[ove_12]] {{.*}} lvalue
// CHECK: |             |-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             |-OpaqueValueExpr [[ove_19]] {{.*}} 'unsigned long long'
// CHECK: |             |-OpaqueValueExpr [[ove_18]] {{.*}} lvalue
// CHECK: |             `-OpaqueValueExpr [[ove_17]] {{.*}} 'unsigned long long'

// CHECK-LABEL: compound_assign_assign
void compound_assign_assign(char *__sized_by(len) p, unsigned long long len) {
  unsigned long long l;
  l = (p+=1, len-=1);
}
// CHECK: | |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_l_1:0x[^ ]+]]
// CHECK: |   `-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK: |     |-DeclRefExpr {{.+}} [[var_l_1]]
// CHECK: |     `-ParenExpr
// CHECK: |       `-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsCheckExpr {{.+}} 'p + 1 <= __builtin_get_pointer_upper_bound(p + 1) && __builtin_get_pointer_lower_bound(p + 1) <= p + 1 && len - 1 <= (char *)__builtin_get_pointer_upper_bound(p + 1) - (char *__bidi_indexable)p + 1'
// CHECK: |         | | |-CompoundAssignOperator {{.+}} ComputeResultTy='char *__single
// CHECK: |         | | | |-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | |   | | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |   | | |     | | | |-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |         | |   | | |     | | | |   |-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |         | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |         | |   | |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |         | |   |   | `-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | |   |     `-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         | |     |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |         | |     |   | `-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} lvalue
// CHECK: |         | |     |   `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |         | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |         | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |         | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |         | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |         | |         |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |         | |             `-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | |-OpaqueValueExpr [[ove_21]]
// CHECK: |         | | `-IntegerLiteral {{.+}} 1
// CHECK: |         | |-OpaqueValueExpr [[ove_20]]
// CHECK: |         | | `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK: |         | |-OpaqueValueExpr [[ove_22]]
// CHECK: |         | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |         | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |         | |   | | | |-OpaqueValueExpr [[ove_23]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |         | |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | |   | | | | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |         | |   | | | | `-AssumptionExpr
// CHECK: |         | |   | | | |   |-OpaqueValueExpr [[ove_24]] {{.*}} 'unsigned long long'
// CHECK: |         | |   | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK: |         | |   | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |         | |   | | | |     | `-OpaqueValueExpr [[ove_24]] {{.*}} 'unsigned long long'
// CHECK: |         | |   | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: |         | |   | | | |       `-IntegerLiteral {{.+}} 0
// CHECK: |         | |   | | |-OpaqueValueExpr [[ove_23]]
// CHECK: |         | |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK: |         | |   | | |   `-OpaqueValueExpr [[ove_20]] {{.*}} lvalue
// CHECK: |         | |   | | `-OpaqueValueExpr [[ove_24]]
// CHECK: |         | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |         | |   | |     `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK: |         | |   | |-OpaqueValueExpr [[ove_23]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |         | |   | `-OpaqueValueExpr [[ove_24]] {{.*}} 'unsigned long long'
// CHECK: |         | |   `-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_27]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |         | |   `-IntegerLiteral {{.+}} 1
// CHECK: |         | |-OpaqueValueExpr [[ove_26]]
// CHECK: |         | | `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK: |         | `-OpaqueValueExpr [[ove_25]]
// CHECK: |         |   `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK: |         |     |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |         |     | `-OpaqueValueExpr [[ove_26]] {{.*}} lvalue
// CHECK: |         |     `-OpaqueValueExpr [[ove_27]] {{.*}} 'unsigned long long'
// CHECK: |         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |           |-CompoundAssignOperator {{.+}} long' ComputeResultTy='unsigned
// CHECK: |           | |-OpaqueValueExpr [[ove_26]] {{.*}} lvalue
// CHECK: |           | `-OpaqueValueExpr [[ove_27]] {{.*}} 'unsigned long long'
// CHECK: |           |-OpaqueValueExpr [[ove_21]] {{.*}} 'int'
// CHECK: |           |-OpaqueValueExpr [[ove_20]] {{.*}} lvalue
// CHECK: |           |-OpaqueValueExpr [[ove_22]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           |-OpaqueValueExpr [[ove_27]] {{.*}} 'unsigned long long'
// CHECK: |           |-OpaqueValueExpr [[ove_26]] {{.*}} lvalue
// CHECK: |           `-OpaqueValueExpr [[ove_25]] {{.*}} 'unsigned long long'

// CHECK-LABEL: assign_init_zero
void assign_init_zero(char *__sized_by(len) p, unsigned long long len) {
  unsigned long long l = (p=0, len=0);
}
// CHECK: | |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_l_2:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK: |           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |           | |-BoundsCheckExpr {{.+}} '0 == 0'
// CHECK: |           | | |-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' '='
// CHECK: |           | | | |-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK: |           | | | `-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |           | | `-BinaryOperator {{.+}} 'int' '=='
// CHECK: |           | |   |-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |           | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |           | |     `-IntegerLiteral {{.+}} 0
// CHECK: |           | |-OpaqueValueExpr [[ove_28]]
// CHECK: |           | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <NullToPointer>
// CHECK: |           | |   `-IntegerLiteral {{.+}} 0
// CHECK: |           | `-OpaqueValueExpr [[ove_29]]
// CHECK: |           |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |           |     `-IntegerLiteral {{.+}} 0
// CHECK: |           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK: |             | |-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK: |             | `-OpaqueValueExpr [[ove_29]] {{.*}} 'unsigned long long'
// CHECK: |             |-OpaqueValueExpr [[ove_28]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |             `-OpaqueValueExpr [[ove_29]] {{.*}} 'unsigned long long'

// CHECK-LABEL: assign_init
void assign_init(char *__sized_by(len) p, unsigned long long len,
                 char *__bidi_indexable ip, unsigned long long ilen) {
  unsigned long long l = (p=ip, len=ilen);
}
// CHECK: | |-ParmVarDecl [[var_p_5:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_5:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_ip:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_ilen:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_l_3:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK: |           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |           | |-BoundsCheckExpr {{.+}} 'ip <= __builtin_get_pointer_upper_bound(ip) && __builtin_get_pointer_lower_bound(ip) <= ip && ilen <= (char *)__builtin_get_pointer_upper_bound(ip) - (char *__bidi_indexable)ip'
// CHECK: |           | | |-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' '='
// CHECK: |           | | | |-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK: |           | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |           | | |   `-OpaqueValueExpr [[ove_30:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |           | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |           | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |   | | | `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |           | |   | |   `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |           | |   |   | `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |   |     `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |           | |     |-OpaqueValueExpr [[ove_31:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |           | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |           | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |           | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |           | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |           | |         |   `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |           | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |           | |             `-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |           | |-OpaqueValueExpr [[ove_30]]
// CHECK: |           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: |           | |   `-DeclRefExpr {{.+}} [[var_ip]]
// CHECK: |           | `-OpaqueValueExpr [[ove_31]]
// CHECK: |           |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |           |     `-DeclRefExpr {{.+}} [[var_ilen]]
// CHECK: |           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK: |             | |-DeclRefExpr {{.+}} [[var_len_5]]
// CHECK: |             | `-OpaqueValueExpr [[ove_31]] {{.*}} 'unsigned long long'
// CHECK: |             |-OpaqueValueExpr [[ove_30]] {{.*}} 'char *__bidi_indexable'
// CHECK: |             `-OpaqueValueExpr [[ove_31]] {{.*}} 'unsigned long long'

// CHECK-LABEL: assign_assign
void assign_assign(char *__sized_by(len) p, unsigned long long len,
                   char *__bidi_indexable ip, unsigned long long ilen) {
  unsigned long long llen = 0;
  char *__sized_by(llen) lp = 0;
  lp = (len = ilen, p=ip);
  llen = 0;
}
// CHECK: | |-ParmVarDecl [[var_p_6:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_6:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_ip_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_ilen_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_llen:0x[^ ]+]]
// CHECK: |   |   |-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |   |   | `-IntegerLiteral {{.+}} 0
// CHECK: |   |   `-DependerDeclsAttr
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_lp_1:0x[^ ]+]]
// CHECK: |   |   `-BoundsCheckExpr {{.+}} 'llen == 0'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(llen)':'char *__single' <NullToPointer>
// CHECK: |   |     | `-IntegerLiteral {{.+}} 0
// CHECK: |   |     `-BinaryOperator {{.+}} 'int' '=='
// CHECK: |   |       |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |   |       | `-DeclRefExpr {{.+}} [[var_llen]]
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |   |         `-IntegerLiteral {{.+}} 0
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr {{.+}} '(len = ilen , p = ip) <= __builtin_get_pointer_upper_bound((len = ilen , p = ip)) && __builtin_get_pointer_lower_bound((len = ilen , p = ip)) <= (len = ilen , p = ip) && 0 <= (char *)__builtin_get_pointer_upper_bound((len = ilen , p = ip)) - (char *__single)(len = ilen , p = ip)'
// CHECK: |   | | |-BinaryOperator {{.+}} 'char *__single __sized_by(llen)':'char *__single' '='
// CHECK: |   | | | |-DeclRefExpr {{.+}} [[var_lp_1]]
// CHECK: |   | | | `-OpaqueValueExpr [[ove_32:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | | |       | | | `-OpaqueValueExpr [[ove_33:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |   | | |       | |   | | | `-OpaqueValueExpr [[ove_34:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   | |   | |     `-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   | |   |   | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |   |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     |-OpaqueValueExpr [[ove_35:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: |   | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |   | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |         |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   | |         |     `-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |         `-CStyleCastExpr {{.+}} 'char *__single' <NoOp>
// CHECK: |   | |           `-OpaqueValueExpr [[ove_32]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK: |   | |-OpaqueValueExpr [[ove_32]]
// CHECK: |   | | `-ParenExpr
// CHECK: |   | |   `-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' ','
// CHECK: |   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |     | |-BoundsCheckExpr {{.+}} 'ip <= __builtin_get_pointer_upper_bound(ip) && __builtin_get_pointer_lower_bound(ip) <= ip && ilen <= (char *)__builtin_get_pointer_upper_bound(ip) - (char *__bidi_indexable)ip'
// CHECK: |   | |     | | |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK: |   | |     | | | |-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK: |   | |     | | | `-OpaqueValueExpr [[ove_33]] {{.*}} 'unsigned long long'
// CHECK: |   | |     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   | |     | |   | | | `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |     | |   | |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   | |     | |   |   | `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   | |     | |   |     `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     | |     |-OpaqueValueExpr [[ove_33]] {{.*}} 'unsigned long long'
// CHECK: |   | |     | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |   | |     | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   | |     | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK: |   | |     | |         | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |     | |         |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   | |     | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK: |   | |     | |             `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |     | |-OpaqueValueExpr [[ove_33]]
// CHECK: |   | |     | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK: |   | |     | |   `-DeclRefExpr {{.+}} [[var_ilen_1]]
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_34]]
// CHECK: |   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: |   | |     |     `-DeclRefExpr {{.+}} [[var_ip_1]]
// CHECK: |   | |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |       |-BinaryOperator {{.+}} 'char *__single __sized_by(len)':'char *__single' '='
// CHECK: |   | |       | |-DeclRefExpr {{.+}} [[var_p_6]]
// CHECK: |   | |       | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |   | |       |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | |       |-OpaqueValueExpr [[ove_33]] {{.*}} 'unsigned long long'
// CHECK: |   | |       `-OpaqueValueExpr [[ove_34]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   | `-OpaqueValueExpr
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: |   |     `-IntegerLiteral {{.+}} 0
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK: |     | |-DeclRefExpr {{.+}} [[var_llen]]
// CHECK: |     | `-OpaqueValueExpr [[ove_35]] {{.*}} 'unsigned long long'

// CHECK-LABEL: for_cond_inc
char for_cond_inc(char *__sized_by(len) p, unsigned long long len) {
  char val = *p;
  for (p++, len--; len > 0; p++, len--) {
    val += *p;
  }
  return val;
}
// CHECK: |-ParmVarDecl [[var_p_7:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len_7:0x[^ ]+]]
// CHECK: | `-DependerDeclsAttr
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_val:0x[^ ]+]]
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK:   |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:   |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   |         | | |-OpaqueValueExpr [[ove_36:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:   |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   |         | | | | `-OpaqueValueExpr [[ove_36]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   |         | | | `-AssumptionExpr
// CHECK:   |         | | |   |-OpaqueValueExpr [[ove_37:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   |         | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK:   |         | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   |         | | |     | `-OpaqueValueExpr [[ove_37]] {{.*}} 'unsigned long long'
// CHECK:   |         | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   |         | | |       `-IntegerLiteral {{.+}} 0
// CHECK:   |         | |-OpaqueValueExpr [[ove_36]]
// CHECK:   |         | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK:   |         | |   `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK:   |         | `-OpaqueValueExpr [[ove_37]]
// CHECK:   |         |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   |         |     `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   |         |-OpaqueValueExpr [[ove_36]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   |         `-OpaqueValueExpr [[ove_37]] {{.*}} 'unsigned long long'
// CHECK:   |-ForStmt
// CHECK:   | |-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK:   | | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | | | |-BoundsCheckExpr {{.+}} 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && len - 1ULL <= (char *)__builtin_get_pointer_upper_bound(p + 1UL) - (char *__bidi_indexable)p + 1UL'
// CHECK:   | | | | |-UnaryOperator {{.+}} postfix '++'
// CHECK:   | | | | | `-OpaqueValueExpr [[ove_38:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   | | | `-OpaqueValueExpr [[ove_39:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   | | |     | | | |-OpaqueValueExpr [[ove_40:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | |     | | | |   |-OpaqueValueExpr [[ove_41:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | | | |   | |   `-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | | | |   |   | `-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   |     `-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |     |-OpaqueValueExpr [[ove_42:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | | | |     |   | `-OpaqueValueExpr [[ove_43:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   | | | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | | | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK:   | | | |         | `-GetBoundExpr {{.+}} upper
// CHECK:   | | | |         |   `-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK:   | | | |             `-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |-OpaqueValueExpr [[ove_38]]
// CHECK:   | | | | `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK:   | | | |-OpaqueValueExpr [[ove_39]]
// CHECK:   | | | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK:   | | | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | | | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | | | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | | | |   | | | |-OpaqueValueExpr [[ove_40]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:   | | | |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   | | | | | `-OpaqueValueExpr [[ove_40]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | | | `-AssumptionExpr
// CHECK:   | | | |   | | | |   |-OpaqueValueExpr [[ove_41]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK:   | | | |   | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   | | | |   | | | |     | `-OpaqueValueExpr [[ove_41]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   | | | |   | | | |       `-IntegerLiteral {{.+}} 0
// CHECK:   | | | |   | | |-OpaqueValueExpr [[ove_40]]
// CHECK:   | | | |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK:   | | | |   | | |   `-OpaqueValueExpr [[ove_38]] {{.*}} lvalue
// CHECK:   | | | |   | | `-OpaqueValueExpr [[ove_41]]
// CHECK:   | | | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   | | | |   | |     `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   | | | |   | |-OpaqueValueExpr [[ove_40]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | `-OpaqueValueExpr [[ove_41]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   `-IntegerLiteral {{.+}} 1
// CHECK:   | | | |-OpaqueValueExpr [[ove_43]]
// CHECK:   | | | | `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   | | | `-OpaqueValueExpr [[ove_42]]
// CHECK:   | | |   `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK:   | | |     |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   | | |     | `-OpaqueValueExpr [[ove_43]] {{.*}} lvalue
// CHECK:   | | |     `-IntegerLiteral {{.+}} 1
// CHECK:   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |   |-UnaryOperator {{.+}} postfix '--'
// CHECK:   | |   | `-OpaqueValueExpr [[ove_43]] {{.*}} lvalue
// CHECK:   | |   |-OpaqueValueExpr [[ove_38]] {{.*}} lvalue
// CHECK:   | |   |-OpaqueValueExpr [[ove_39]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |   |-OpaqueValueExpr [[ove_43]] {{.*}} lvalue
// CHECK:   | |   `-OpaqueValueExpr [[ove_42]] {{.*}} 'unsigned long long'
// CHECK:   | |-BinaryOperator {{.+}} 'int' '>'
// CHECK:   | | |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   | | | `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   | |   `-IntegerLiteral {{.+}} 0
// CHECK:   | |-BinaryOperator {{.+}} 'unsigned long long' ','
// CHECK:   | | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | | | |-BoundsCheckExpr {{.+}} 'p + 1UL <= __builtin_get_pointer_upper_bound(p + 1UL) && __builtin_get_pointer_lower_bound(p + 1UL) <= p + 1UL && len - 1ULL <= (char *)__builtin_get_pointer_upper_bound(p + 1UL) - (char *__bidi_indexable)p + 1UL'
// CHECK:   | | | | |-UnaryOperator {{.+}} postfix '++'
// CHECK:   | | | | | `-OpaqueValueExpr [[ove_44:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   | | | `-OpaqueValueExpr [[ove_45:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   | | |     | | | |-OpaqueValueExpr [[ove_46:0x[^ ]+]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | |     | | | |   |-OpaqueValueExpr [[ove_47:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | | | |   | |   `-OpaqueValueExpr [[ove_45]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | | | |   |   | `-OpaqueValueExpr [[ove_45]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   |     `-OpaqueValueExpr [[ove_45]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | | | |     |-OpaqueValueExpr [[ove_48:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | | | |     |   | `-OpaqueValueExpr [[ove_49:0x[^ ]+]] {{.*}} lvalue
// CHECK:   | | | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   | | | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | | | |         |-CStyleCastExpr {{.+}} 'char *' <NoOp>
// CHECK:   | | | |         | `-GetBoundExpr {{.+}} upper
// CHECK:   | | | |         |   `-OpaqueValueExpr [[ove_45]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// CHECK:   | | | |             `-OpaqueValueExpr [[ove_45]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | | |-OpaqueValueExpr [[ove_44]]
// CHECK:   | | | | `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK:   | | | |-OpaqueValueExpr [[ove_45]]
// CHECK:   | | | | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK:   | | | |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | | | |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | | | |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | | | |   | | | |-OpaqueValueExpr [[ove_46]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:   | | | |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | | | |   | | | | | `-OpaqueValueExpr [[ove_46]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | | | | `-AssumptionExpr
// CHECK:   | | | |   | | | |   |-OpaqueValueExpr [[ove_47]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | | |   `-BinaryOperator {{.+}} 'int' '>='
// CHECK:   | | | |   | | | |     |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   | | | |   | | | |     | `-OpaqueValueExpr [[ove_47]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   | | | |     `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK:   | | | |   | | | |       `-IntegerLiteral {{.+}} 0
// CHECK:   | | | |   | | |-OpaqueValueExpr [[ove_46]]
// CHECK:   | | | |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(len)':'char *__single' <LValueToRValue>
// CHECK:   | | | |   | | |   `-OpaqueValueExpr [[ove_44]] {{.*}} lvalue
// CHECK:   | | | |   | | `-OpaqueValueExpr [[ove_47]]
// CHECK:   | | | |   | |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   | | | |   | |     `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   | | | |   | |-OpaqueValueExpr [[ove_46]] {{.*}} 'char *__single __sized_by(len)':'char *__single'
// CHECK:   | | | |   | `-OpaqueValueExpr [[ove_47]] {{.*}} 'unsigned long long'
// CHECK:   | | | |   `-IntegerLiteral {{.+}} 1
// CHECK:   | | | |-OpaqueValueExpr [[ove_49]]
// CHECK:   | | | | `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:   | | | `-OpaqueValueExpr [[ove_48]]
// CHECK:   | | |   `-BinaryOperator {{.+}} 'unsigned long long' '-'
// CHECK:   | | |     |-ImplicitCastExpr {{.+}} 'unsigned long long' <LValueToRValue>
// CHECK:   | | |     | `-OpaqueValueExpr [[ove_49]] {{.*}} lvalue
// CHECK:   | | |     `-IntegerLiteral {{.+}} 1
// CHECK:   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |   |-UnaryOperator {{.+}} postfix '--'
// CHECK:   | |   | `-OpaqueValueExpr [[ove_49]] {{.*}} lvalue
// CHECK:   | `-CompoundStmt

