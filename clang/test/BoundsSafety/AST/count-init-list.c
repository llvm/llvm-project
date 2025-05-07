

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Foo {
    int len;
    int *__counted_by(len) ptr;
};

void Test(void) {
    int *p;
    struct Foo f = { 10, p };
}

// CHECK-LABEL: Test 'void (void)'
// CHECK: CompoundStmt
// CHECK: |-DeclStmt
// CHECK: | `-VarDecl [[var_p:0x[^ ]+]]
// CHECK: `-DeclStmt
// CHECK:   `-VarDecl [[var_f:0x[^ ]+]]
// CHECK:     `-BoundsCheckExpr
// CHECK:       |-InitListExpr
// CHECK:       | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-GetBoundExpr {{.+}} upper
// CHECK:       |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove]]
// CHECK:       | `-IntegerLiteral {{.+}} 10
// CHECK:       `-OpaqueValueExpr [[ove_1]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_p]]
