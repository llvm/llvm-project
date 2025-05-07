
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | `-DependerDeclsAttr
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | | | |-UnaryOperator {{.+}} cannot overflow
// CHECK:     | | | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:     | | | |   `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   | | |     | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK:     | |   | | |     | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |     | | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:     | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_1]]
// CHECK:     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:     |   | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK:     |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     |   | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK:     |   | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     |   | |-OpaqueValueExpr [[ove_2]]
// CHECK:     |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)':'int *__single' <LValueToRValue>
// CHECK:     |   | |   `-DeclRefExpr {{.+}} [[var_buf]]
// CHECK:     |   | `-OpaqueValueExpr [[ove_3]]
// CHECK:     |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   |     `-UnaryOperator {{.+}} cannot overflow
// CHECK:     |   |       `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:     |   |         `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:     |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(*len)':'int *__single'
// CHECK:     |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     `-OpaqueValueExpr [[ove]]
// CHECK:       `-IntegerLiteral {{.+}} 42
void foo(int *__counted_by(*len) buf, int *len) {
  *len = 42;
}

// CHECK: FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_len_1:0x[^ ]+]]
// CHECK:   |   `-IntegerLiteral {{.+}} 11
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(*len), int *__single)' <FunctionToPointerDecay>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'long'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |       | `-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'long'
// CHECK:     | |-OpaqueValueExpr [[ove_4]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:     | |-OpaqueValueExpr [[ove_5]]
// CHECK:     | | `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:     | `-OpaqueValueExpr [[ove_6]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |       `-UnaryOperator {{.+}} cannot overflow
// CHECK:     |         `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_6]] {{.*}} 'long'
void bar() {
  int arr[10];
  int len = 11;
  foo(arr, &len);
}
