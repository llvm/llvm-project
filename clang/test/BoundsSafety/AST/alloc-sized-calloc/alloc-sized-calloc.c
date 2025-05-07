

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include "mock-header.h"

int foo() {
    int cnt = 10;
    int siz = sizeof(int);
    int *ptr1 = my_calloc(cnt, siz);
    int *__bidi_indexable ptr2;
    ptr2 = my_calloc(cnt, siz);
    return ptr2[cnt-1];
}

// CHECK: {{^}}TranslationUnitDecl
// CHECK: {{^}}|-FunctionDecl [[func_my_calloc:0x[^ ]+]] {{.+}} my_calloc
// CHECK: {{^}}| |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: {{^}}| `-AllocSizeAttr
// CHECK: {{^}}`-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_cnt:0x[^ ]+]]
// CHECK: {{^}}    |   `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_siz:0x[^ ]+]]
// CHECK: {{^}}    |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}    |     `-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_ptr1:0x[^ ]+]]
// CHECK: {{^}}    |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: {{^}}    |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |       | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}    |       | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *'
// CHECK: {{^}}    |       | | |   |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |       | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |       | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}    |       | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |       | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |       | | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'void *'
// CHECK: {{^}}    |       | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |       | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}    |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |       | |   `-DeclRefExpr {{.+}} [[var_cnt]]
// CHECK: {{^}}    |       | |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}    |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |       | |   `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}    |       | |-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |       | | `-CallExpr
// CHECK: {{^}}    |       | |   |-ImplicitCastExpr {{.+}} 'void *(*__single)(int, int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |       | |   | `-DeclRefExpr {{.+}} [[func_my_calloc]]
// CHECK: {{^}}    |       | |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |       | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}    |       | `-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}    |       |   `-BinaryOperator {{.+}} 'int' '*'
// CHECK: {{^}}    |       |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |       |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}    |       |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}    |       |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}    |       |-OpaqueValueExpr [[ove]] {{.*}} 'void *'
// CHECK: {{^}}    |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_ptr2:0x[^ ]+]]
// CHECK: {{^}}    |-BinaryOperator {{.+}} 'int *__bidi_indexable' '='
// CHECK: {{^}}    | |-DeclRefExpr {{.+}} [[var_ptr2]]
// CHECK: {{^}}    | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_cnt]]
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}    |     | | `-CallExpr
// CHECK: {{^}}    |     | |   |-ImplicitCastExpr {{.+}} 'void *(*__single)(int, int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     | |   | `-DeclRefExpr {{.+}} [[func_my_calloc]]
// CHECK: {{^}}    |     | |   |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: {{^}}    |     | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}    |     |   `-BinaryOperator {{.+}} 'int' '*'
// CHECK: {{^}}    |     |     |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'void *'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}    `-ReturnStmt
// CHECK: {{^}}      `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}        `-ArraySubscriptExpr
// CHECK: {{^}}          |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}          | `-DeclRefExpr {{.+}} [[var_ptr2]]
// CHECK: {{^}}          `-BinaryOperator {{.+}} 'int' '-'
// CHECK: {{^}}            |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}            | `-DeclRefExpr {{.+}} [[var_cnt]]
// CHECK: {{^}}            `-IntegerLiteral {{.+}} 1
