

// RUN: %clang_cc1 -ast-dump -fbounds-safety -isystem %S/Inputs/system-count-return %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -isystem %S/Inputs/system-count-return %s 2>&1 | FileCheck %s

#include <mock-header.h>
// CHECK: FunctionDecl [[func_alloc_sized_by:0x[^ ]+]] {{.+}} alloc_sized_by
// CHECK: FunctionDecl [[func_alloc_attributed:0x[^ ]+]] {{.+}} alloc_attributed

int Test() {
    int len = 16;
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_len_2:0x[^ ]+]]
// CHECK: {{^}}    |   `-IntegerLiteral {{.+}} 16

    int *bufAuto = alloc_sized_by(len);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_bufAuto:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} [[func_alloc_sized_by]]
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(len)':'int *__single'

    int *__bidi_indexable bufBound = alloc_attributed(sizeof(int) * 10);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_bufBound:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}    |     | |   `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: {{^}}    |     | |     |-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}    |     | |     `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}    |     | |       `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'int *(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} [[func_alloc_attributed]]
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *'

    return bufBound[10];
}
