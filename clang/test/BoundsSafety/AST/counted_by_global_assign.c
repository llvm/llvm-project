

// FileCheck doesn't seem to be able to handle too many slashes in a line
// XFAIL: *

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

int glen;
int *__counted_by(glen) gptr;
// CHECK: |-VarDecl [[var_len:0x[^ ]+]]
// CHECK: {{^ *}}DependerDeclsAttr
// CHECK: |-VarDecl [[var_ptr:0x[^ ]+]]

// CHECK-LABEL: test
void test(int *__bidi_indexable arg) {
  glen = 0;
  gptr = arg;
}
// CHECK: |-ParmVarDecl [[var_arg:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK:   | | | |-DeclRefExpr {{.+}} [[var_len]]
// CHECK:   | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-GetBoundExpr {{.+}} lower
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | |-IntegerLiteral {{.+}} 0
// CHECK:   | |     | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:   | |       | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | |         |-GetBoundExpr {{.+}} upper
// CHECK:   | |         | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |         `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |           `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove]]
// CHECK:   | | `-IntegerLiteral {{.+}} 0
// CHECK:   | `-OpaqueValueExpr [[ove_1]]
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:   |     `-DeclRefExpr {{.+}} [[var_arg]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK:     | |-DeclRefExpr {{.+}} [[var_ptr]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
