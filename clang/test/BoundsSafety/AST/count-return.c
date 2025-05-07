

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

int *__sized_by(len) alloc(int len);
// CHECK: FunctionDecl [[func_alloc:0x[^ ]+]] {{.+}} alloc

int *__counted_by(4) noproto(); // non-prototype function
// CHECK: FunctionDecl [[func_noproto:0x[^ ]+]] {{.+}} noproto 'int *__single __counted_by(4)()'

int Test() {
    int len = 16;
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_len_1:0x[^ ]+]]

    int *bufAuto = alloc(len);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_bufAuto:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} [[func_alloc]]
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(len)':'int *__single'

    int *__bidi_indexable bufBound = alloc(sizeof(int) * 10);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_bufBound:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}    |     | |   `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: {{^}}    |     | |     |-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}    |     | |     `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}    |     | |       `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} [[func_alloc]]
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(len)':'int *__single'

    int *buf4 = noproto();
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_buf4:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __counted_by(4)':'int *__single'
// CHECK: {{^}}    |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}    |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(4)':'int *__single'
// CHECK: {{^}}    |     | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}    |     | | `-CallExpr
// CHECK: {{^}}    |     | |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(4)(*__single)()' <FunctionToPointerDecay>
// CHECK: {{^}}    |     | |     `-DeclRefExpr {{.+}} [[func_noproto]]
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}    |     |   `-IntegerLiteral {{.+}} 4
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(4)':'int *__single'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'

    return buf4[3] + bufBound[9];
// CHECK: {{^}}    `-ReturnStmt
// CHECK: {{^}}      `-BinaryOperator {{.+}} 'int' '+'
// CHECK: {{^}}        |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}        | `-ArraySubscriptExpr
// CHECK: {{^}}        |   |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}        |   | `-DeclRefExpr {{.+}} [[var_buf4]]
// CHECK: {{^}}        |   `-IntegerLiteral {{.+}} 3
// CHECK: {{^}}        `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}          `-ArraySubscriptExpr
// CHECK: {{^}}            |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}            | `-DeclRefExpr {{.+}} [[var_bufBound]]
// CHECK: {{^}}            `-IntegerLiteral {{.+}} 9
}
