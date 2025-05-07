

// RUN: not %clang_cc1 -fbounds-safety -ast-dump %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: not %clang_cc1 -fbounds-safety -ast-dump %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s


#include <ptrcheck.h>

typedef struct {
    int len;
    int *__counted_by(len) buf;
} S;

void Foo(int *__counted_by(len) buf, int len) {}
void Bar(int *__sized_by(siz) buf, int siz) {}
// CHECK: FunctionDecl [[func_Foo:0x[^ ]+]] {{.+}} Foo
// CHECK: FunctionDecl [[func_Bar:0x[^ ]+]] {{.+}} Bar

void Test(void) {
// CHECK-LABEL: FunctionDecl {{.+}} Test
    int value = 0;
// CHECK: VarDecl [[var_value:0x[^ ]+]] {{.+}} value
    int *__single p = &value;
// CHECK: VarDecl [[var_p:0x[^ ]+]] {{.+}} p
    int *q = &value;
// CHECK: VarDecl [[var_q:0x[^ ]+]] {{.+}} q
    S s;
// CHECK: VarDecl [[var_s:0x[^ ]+]] {{.+}} s
Lassign:
    s.buf = &value;
    s.len = 1;
// CHECK-LABEL: Lassign
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   |-BoundsCheckExpr
// CHECK: |   |   | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK: |   |   | | |-MemberExpr {{.+}} .buf
// CHECK: |   |   | | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   |   |   |-GetBoundExpr {{.+}} lower
// CHECK: |   |   |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   |     | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   |     |   |-GetBoundExpr {{.+}} upper
// CHECK: |   |   |     |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |   |   |-OpaqueValueExpr [[ove]]
// CHECK: |   |   | `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   |   |   `-DeclRefExpr {{.+}} [[var_value]]
// CHECK: |   |   `-OpaqueValueExpr [[ove_1]]
// CHECK: |   |     `-IntegerLiteral {{.+}} 1
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   | |-BinaryOperator {{.+}} 'int' '='
// CHECK: |   | | |-MemberExpr {{.+}} .len
// CHECK: |   | | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: |   | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'

Lfoo_sbuf_0:
    Foo(s.buf, 0);
// CHECK-LABEL: Lfoo_sbuf_0
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | | | |     | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | | | |     | | |     `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   |   | | | |     | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: |   |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_4]]
// CHECK: |   |   | |   | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_5]]
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   | |   | |   `-MemberExpr {{.+}} .len
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} lvalue
// CHECK: |   |   | |   | `-OpaqueValueExpr [[ove_3]]
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   |   | |   |     `-MemberExpr {{.+}} .buf
// CHECK: |   |   | |   |       `-OpaqueValueExpr [[ove_4]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_4]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: |   |   | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | `-OpaqueValueExpr [[ove_6]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 0
// CHECK: |   |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'

Lfoo_sbuf_1:
    Foo(s.buf, 1);
// CHECK-LABEL: Lfoo_sbuf_1
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | | | |     | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | | | |     | | |     `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   |   | | | |     | | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_7]]
// CHECK: |   |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_9]]
// CHECK: |   |   | |   | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_10]]
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   | |   | |   `-MemberExpr {{.+}} .len
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |   |   | |   | `-OpaqueValueExpr [[ove_8]]
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   |   | |   |     `-MemberExpr {{.+}} .buf
// CHECK: |   |   | |   |       `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |   |   | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | `-OpaqueValueExpr [[ove_11]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 1
// CHECK: |   |   |-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'

Lfoo_sbuf_5:
    Foo(s.buf, 5);
// CHECK-LABEL: Lfoo_sbuf_5
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | | | |     | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | | | |     | | |     `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   |   | | | |     | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_12]]
// CHECK: |   |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_14]]
// CHECK: |   |   | |   | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_15]]
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   | |   | |   `-MemberExpr {{.+}} .len
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |   |   | |   | `-OpaqueValueExpr [[ove_13]]
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   |   | |   |     `-MemberExpr {{.+}} .buf
// CHECK: |   |   | |   |       `-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_14]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: |   |   | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | `-OpaqueValueExpr [[ove_16]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 5
// CHECK: |   |   |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'

Lfoo_p_0:
    Foo(p, 0);
// CHECK-LABEL: Lfoo_p_0
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |       `-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |   |   `-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |     |   `-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_17]]
// CHECK: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: |   |   | |   `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |   |   | `-OpaqueValueExpr [[ove_18]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 0
// CHECK: |   |   |-OpaqueValueExpr [[ove_17]] {{.*}} 'int *__single'
// CHECK: |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int'

Lfoo_p_1:
    Foo(p, 1);
// CHECK-LABEL: Lfoo_p_1
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |       `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |   |   `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |     |   `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_19]]
// CHECK: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: |   |   | |   `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |   |   | `-OpaqueValueExpr [[ove_20]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 1
// CHECK: |   |   |-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single'
// CHECK: |   |   `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'

Lfoo_q_5:
    Foo(q, 5);
// CHECK-LABEL: Lfoo_q_5
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Foo]]
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |   `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_21]]
// CHECK: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: |   |   | |   `-DeclRefExpr {{.+}} [[var_q]]
// CHECK: |   |   | `-OpaqueValueExpr [[ove_22]]
// CHECK: |   |   |   `-IntegerLiteral {{.+}} 5
// CHECK: |   |   |-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'

Lbar_sbuf_slen:
    Bar(s.buf, s.len * sizeof(int));
// CHECK-LABEL: Lbar_sbuf_slen
// CHECK: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |-BoundsCheckExpr
// CHECK: |   |   | | |-CallExpr
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(siz), int)' <FunctionToPointerDecay>
// CHECK: |   |   | | | | `-DeclRefExpr {{.+}} [[func_Bar]]
// CHECK: |   |   | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(siz)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   |   | | | | `-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | | | |     | | |-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | | | |     | | |     `-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} lvalue
// CHECK: |   |   | | | |     | | | `-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | | `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |   |   | |   |   |   `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |   | |     | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: |   |   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |   | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |   |   | |     |   |   `-GetBoundExpr {{.+}} upper
// CHECK: |   |   | |     |   |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |   |   | |     |       `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |   | |       |-IntegerLiteral {{.+}} 0
// CHECK: |   |   | |       `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: |   |   | |-OpaqueValueExpr [[ove_23]]
// CHECK: |   |   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |   |   | |   | | |-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |   |   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   |   | |   | | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | |   | | | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_25]]
// CHECK: |   |   | |   | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   | |   | |-OpaqueValueExpr [[ove_26]]
// CHECK: |   |   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   | |   | |   `-MemberExpr {{.+}} .len
// CHECK: |   |   | |   | |     `-OpaqueValueExpr [[ove_25]] {{.*}} lvalue
// CHECK: |   |   | |   | `-OpaqueValueExpr [[ove_24]]
// CHECK: |   |   | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |   |   | |   |     `-MemberExpr {{.+}} .buf
// CHECK: |   |   | |   |       `-OpaqueValueExpr [[ove_25]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_25]] {{.*}} lvalue
// CHECK: |   |   | |   |-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK: |   |   | |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |   |   | `-OpaqueValueExpr [[ove_27]]
// CHECK: |   |   |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |   |   |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: |   |   |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |   |   |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   |       |   `-MemberExpr {{.+}} .len
// CHECK: |   |   |       |     `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |   |   |       `-UnaryExprOrTypeTraitExpr
// CHECK: |   |   |-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   |   `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'

Lbar_sbuf_11:
    Bar(s.buf, 11 * sizeof(int));
// CHECK-LABEL: Lbar_sbuf_11
// CHECK: |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |-BoundsCheckExpr
// CHECK: |       | | |-CallExpr
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(siz), int)' <FunctionToPointerDecay>
// CHECK: |       | | | | `-DeclRefExpr {{.+}} [[func_Bar]]
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(siz)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |       | | | | `-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | | | |     | | |-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |       | | | |     | | |     `-OpaqueValueExpr [[ove_30:0x[^ ]+]] {{.*}} lvalue
// CHECK: |       | | | |     | | | `-OpaqueValueExpr [[ove_31:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | | `-OpaqueValueExpr [[ove_32:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |       | |   | |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |       | |   |   |   `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       | |     | | `-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
// CHECK: |       | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |       | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |       | |     |   |   `-GetBoundExpr {{.+}} upper
// CHECK: |       | |     |   |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |       | |     |       `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |       |-IntegerLiteral {{.+}} 0
// CHECK: |       | |       `-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
// CHECK: |       | |-OpaqueValueExpr [[ove_28]]
// CHECK: |       | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: |       | |   | | |-OpaqueValueExpr [[ove_29]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |       | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: |       | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | | `-OpaqueValueExpr [[ove_29]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK: |       | |   | |-OpaqueValueExpr [[ove_30]]
// CHECK: |       | |   | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: |       | |   | |-OpaqueValueExpr [[ove_31]]
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |   | |   `-MemberExpr {{.+}} .len
// CHECK: |       | |   | |     `-OpaqueValueExpr [[ove_30]] {{.*}} lvalue
// CHECK: |       | |   | `-OpaqueValueExpr [[ove_29]]
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: |       | |   |     `-MemberExpr {{.+}} .buf
// CHECK: |       | |   |       `-OpaqueValueExpr [[ove_30]] {{.*}} lvalue
// CHECK: |       | |   |-OpaqueValueExpr [[ove_30]] {{.*}} lvalue
// CHECK: |       | |   |-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK: |       | |   `-OpaqueValueExpr [[ove_29]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: |       | `-OpaqueValueExpr [[ove_32]]
// CHECK: |       |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: |       |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: |       |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |       |       | `-IntegerLiteral {{.+}} 11
// CHECK: |       |       `-UnaryExprOrTypeTraitExpr
// CHECK: |       |-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__bidi_indexable'
// CHECK: |       `-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
}

// CHECK-LABEL: TestError
void TestError(void) {
    int value = 0;
    int *__single p = &value;
    // expected-error@+1{{passing 'int *__single' to parameter 'buf' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 5 always fails}}
    Foo(p, 5);
}
