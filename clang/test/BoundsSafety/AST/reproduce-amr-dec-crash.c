
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

// CHECK: FunctionDecl [[func_conv:0x[^ ]+]] {{.+}} conv
void conv(short x[__counted_by(m)], short y[__counted_by(m)], short m);

void test() {
  short x[16];
  short y[16];
  conv(x, y, 16);
}
// CHECK-LABEL: test 'void ()'
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_x_1:0x[^ ]+]]
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_y_1:0x[^ ]+]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-BoundsCheckExpr
// CHECK:     | | | |-CallExpr
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(short *__single __counted_by(m), short *__single __counted_by(m), short)' <FunctionToPointerDecay>
// CHECK:     | | | | | `-DeclRefExpr {{.+}} [[func_conv]]
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'short *__single __counted_by(m)':'short *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'short *__single __counted_by(m)':'short *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'short'
// CHECK:     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |   | | |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |   | | `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |   |   |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | | |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |   |   `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |     | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'short'
// CHECK:     | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | | |     |   |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK:     | | |     |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |     |   `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | | |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |       |-IntegerLiteral {{.+}} 0
// CHECK:     | | |       `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK:     | | |         `-OpaqueValueExpr [[ove_2]] {{.*}} 'short'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |   | | `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |     | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'short'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'short *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK:     | |         `-OpaqueValueExpr [[ove_2]] {{.*}} 'short'
// CHECK:     | |-OpaqueValueExpr [[ove]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'short *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_x_1]]
// CHECK:     | |-OpaqueValueExpr [[ove_1]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'short *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_y_1]]
// CHECK:     | `-OpaqueValueExpr [[ove_2]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'short' <IntegralCast>
// CHECK:     |     `-IntegerLiteral {{.+}} 16
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} 'short *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_1]] {{.*}} 'short *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_2]] {{.*}} 'short'
