

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Check the GNU extension to the conditional operator.

// CHECK: FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: `-ParmVarDecl [[var_p:0x[^ ]+]] {{.+}} p 'void *__single __sized_by(16)':'void *__single'
void foo(void *__sized_by(16) p);

// CHECK: FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_buf:0x[^ ]+]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(void *__single __sized_by(16))' <FunctionToPointerDecay>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(16)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | |         |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'void *'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'long'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |   |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |       `-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |       | `-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
// CHECK:     | |-OpaqueValueExpr [[ove]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-BinaryConditionalOperator {{.+}} 'unsigned char *__bidi_indexable'
// CHECK:     | |     |-CStyleCastExpr {{.+}} 'void *' <NullToPointer>
// CHECK:     | |     | `-IntegerLiteral {{.+}} 0
// CHECK:     | |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <NullToPointer>
// CHECK:     | |     | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *'
// CHECK:     | |     `-ImplicitCastExpr {{.+}} 'unsigned char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |       `-DeclRefExpr {{.+}} [[var_buf]]
// CHECK:     | `-OpaqueValueExpr [[ove_2]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     `-IntegerLiteral {{.+}} 16
// CHECK:     |-OpaqueValueExpr [[ove]] {{.*}} 'void *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
void bar(void) {
  unsigned char buf[16];
  foo((void *)0 ?: buf);
}

// CHECK: FunctionDecl [[func_returns_a_pointer:0x[^ ]+]] {{.+}} returns_a_pointer
void *__bidi_indexable returns_a_pointer(void) {
  return 0;
}

// CHECK: FunctionDecl [[func_baz:0x[^ ]+]] {{.+}} baz
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsCheckExpr
// CHECK:     | | |-CallExpr
// CHECK:     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(void *__single __sized_by(16))' <FunctionToPointerDecay>
// CHECK:     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(16)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | |       |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |   | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |   |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | |   |   |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |     | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'long'
// CHECK:     | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |   |   `-GetBoundExpr {{.+}} upper
// CHECK:     | |     |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     | |     |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |       | `-IntegerLiteral {{.+}} 0
// CHECK:     | |       `-OpaqueValueExpr [[ove_5]] {{.*}} 'long'
// CHECK:     | |-OpaqueValueExpr [[ove_3]]
// CHECK:     | | `-BinaryConditionalOperator {{.+}} 'void *__bidi_indexable'
// CHECK:     | |   |-CallExpr
// CHECK:     | |   | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable(*__single)(void)' <FunctionToPointerDecay>
// CHECK:     | |   |   `-DeclRefExpr {{.+}} [[func_returns_a_pointer]]
// CHECK:     | |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <NullToPointer>
// CHECK:     | |     `-IntegerLiteral {{.+}} 0
// CHECK:     | `-OpaqueValueExpr [[ove_5]]
// CHECK:     |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     `-IntegerLiteral {{.+}} 16
// CHECK:     |-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_5]] {{.*}} 'long'
void baz(void) {
  foo(returns_a_pointer() ?: 0);
}

int bounds_safety_func(int * __counted_by(b) a, int b);
// CHECK: FunctionDecl [[func_bounds_safety_func:0x[^ ]+]] {{.+}} bounds_safety_func
// CHECK: |-ParmVarDecl [[var_a:0x[^ ]+]]
// CHECK: `-ParmVarDecl [[var_b:0x[^ ]+]]
// CHECK:   `-DependerDeclsAttr

int eval_count_arg(int * __bidi_indexable a, int b) {
  return bounds_safety_func(a, b ?: 0);
}

// CHECK: FunctionDecl [[func_eval_count_arg:0x[^ ]+]] {{.+}} eval_count_arg
// CHECK: |-ParmVarDecl [[var_a_1:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_b_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:       | |-BoundsCheckExpr {{.+}} 'a <= __builtin_get_pointer_upper_bound(a) && __builtin_get_pointer_lower_bound(a) <= a && b ?: 0 <= __builtin_get_pointer_upper_bound(a) - a && 0 <= b ?: 0'
// CHECK:       | | |-CallExpr
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int (*__single)(int *__single __counted_by(b), int)' <FunctionToPointerDecay>
// CHECK:       | | | | `-DeclRefExpr {{.+}} [[func_bounds_safety_func]]
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(b)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK:       | | |     |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |   | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:       | |   | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:       | |   |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |   |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       | |     | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:       | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK:       | |     |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |     |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |       |-IntegerLiteral {{.+}} 0
// CHECK:       | |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:       | |-OpaqueValueExpr [[ove_6]]
// CHECK:       | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:       | |   `-DeclRefExpr {{.+}} [[var_a_1]]
// CHECK:       | `-OpaqueValueExpr [[ove_7]]
// CHECK:       |   `-BinaryConditionalOperator {{.+}} 'int'
// CHECK:       |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       |     | `-DeclRefExpr {{.+}} [[var_b_1]]
// CHECK:       |     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK:       |     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK:       |     `-IntegerLiteral {{.+}} 0
// CHECK:       |-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:       `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'

