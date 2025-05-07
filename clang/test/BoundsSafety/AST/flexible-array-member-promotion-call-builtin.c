// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// Make sure that calling a builtin and function has the same checks.

// CHECK: FunctionDecl [[func_my_memset:0x[^ ]+]] {{.+}} my_memset
void *__sized_by(len) my_memset(void *__sized_by(len) b, int c, unsigned long len);

typedef struct {
  int count;
  int elems[__counted_by(count)];
} flex_t;

// CHECK: FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: |-ParmVarDecl [[var_flex:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:     | | |   | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | |   | |       | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'flex_t *__single'
// CHECK:     | | |   | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | |   | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK:     | | |   `-AssumptionExpr
// CHECK:     | | |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long'
// CHECK:     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK:     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |       | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long'
// CHECK:     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |         `-IntegerLiteral {{.+}} 0
// CHECK:     | |-OpaqueValueExpr [[ove_1]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'flex_t *__bidi_indexable'
// CHECK:     | |     | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | |     | | | | `-MemberExpr {{.+}} ->elems
// CHECK:     | |     | | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |     | | |   `-MemberExpr {{.+}} ->count
// CHECK:     | |     | | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | `-OpaqueValueExpr [[ove_2]]
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     | |     |     `-DeclRefExpr {{.+}} [[var_flex]]
// CHECK:     | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'flex_t *__single'
// CHECK:     | |-OpaqueValueExpr [[ove_3]]
// CHECK:     | | `-IntegerLiteral {{.+}} 0
// CHECK:     | |-OpaqueValueExpr [[ove_4]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:     | |     `-DeclRefExpr {{.+}} [[var_size]]
// CHECK:     | `-OpaqueValueExpr [[ove]]
// CHECK:     |   `-BoundsCheckExpr {{.+}} 'flex <= __builtin_get_pointer_upper_bound(flex) && __builtin_get_pointer_lower_bound(flex) <= flex && size <= (char *)__builtin_get_pointer_upper_bound(flex) - (char *)flex'
// CHECK:     |     |-CallExpr
// CHECK:     |     | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), int, unsigned long)' <BuiltinFnToFnPtr>
// CHECK:     |     | | `-DeclRefExpr {{.+}}
// CHECK:     |     | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:     |     | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |     | |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long'
// CHECK:     |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |       | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       | |   `-GetBoundExpr {{.+}} upper
// CHECK:     |       | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |       |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       |   | `-GetBoundExpr {{.+}} lower
// CHECK:     |       |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |         |-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long'
// CHECK:     |         `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:     |           `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     |             |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |             | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     |             |   `-GetBoundExpr {{.+}} upper
// CHECK:     |             |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |             `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |               `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     |                 `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'unsigned long'
// CHECK:     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
void foo(flex_t *flex, unsigned size) {
  __builtin_memset(flex, 0, size);
}

// CHECK: FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK: |-ParmVarDecl [[var_flex_1:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:     | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:     | | |   | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:     | | |   | |       | | |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'flex_t *__single'
// CHECK:     | | |   | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | |   | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:     | | |   `-AssumptionExpr
// CHECK:     | | |     |-OpaqueValueExpr [[ove_9]] {{.*}} 'unsigned long'
// CHECK:     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK:     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |       | `-OpaqueValueExpr [[ove_9]] {{.*}} 'unsigned long'
// CHECK:     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |         `-IntegerLiteral {{.+}} 0
// CHECK:     | |-OpaqueValueExpr [[ove_6]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:     | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'flex_t *__bidi_indexable'
// CHECK:     | |     | | |-OpaqueValueExpr [[ove_7]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:     | |     | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | |     | | | | `-MemberExpr {{.+}} ->elems
// CHECK:     | |     | | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     | |     | | |   `-MemberExpr {{.+}} ->count
// CHECK:     | |     | | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'flex_t *__single'
// CHECK:     | |     | `-OpaqueValueExpr [[ove_7]]
// CHECK:     | |     |   `-ImplicitCastExpr {{.+}} 'flex_t *__single' <LValueToRValue>
// CHECK:     | |     |     `-DeclRefExpr {{.+}} [[var_flex_1]]
// CHECK:     | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'flex_t *__single'
// CHECK:     | |-OpaqueValueExpr [[ove_8]]
// CHECK:     | | `-IntegerLiteral {{.+}} 0
// CHECK:     | |-OpaqueValueExpr [[ove_9]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK:     | |     `-DeclRefExpr {{.+}} [[var_size_1]]
// CHECK:     | `-OpaqueValueExpr [[ove_5]]
// CHECK:     |   `-BoundsCheckExpr {{.+}} 'flex <= __builtin_get_pointer_upper_bound(flex) && __builtin_get_pointer_lower_bound(flex) <= flex && size <= (char *)__builtin_get_pointer_upper_bound(flex) - (char *)flex'
// CHECK:     |     |-CallExpr
// CHECK:     |     | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(len)(*__single)(void *__single __sized_by(len), int, unsigned long)' <FunctionToPointerDecay>
// CHECK:     |     | | `-DeclRefExpr {{.+}} [[func_my_memset]]
// CHECK:     |     | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(len)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:     |     | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |     | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK:     |     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'unsigned long'
// CHECK:     |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |       | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       | |   `-GetBoundExpr {{.+}} upper
// CHECK:     |       | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |       |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       |   | `-GetBoundExpr {{.+}} lower
// CHECK:     |       |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:     |       |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |         |-OpaqueValueExpr [[ove_9]] {{.*}} 'unsigned long'
// CHECK:     |         `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:     |           `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     |             |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |             | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     |             |   `-GetBoundExpr {{.+}} upper
// CHECK:     |             |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |             `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |               `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     |                 `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK:     |-OpaqueValueExpr [[ove_9]] {{.*}} 'unsigned long'
// CHECK:     `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
void bar(flex_t *flex, unsigned size) {
  my_memset(flex, 0, size);
}
