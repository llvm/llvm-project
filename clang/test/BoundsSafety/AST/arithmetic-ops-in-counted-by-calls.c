
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>

void param_with_count(int *__counted_by(len - 2) buf, int len);
// CHECK: |-FunctionDecl [[func_param_with_count:0x[^ ]+]] {{.+}} param_with_count
// CHECK: | |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: | `-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: |   `-DependerDeclsAttr


void *__sized_by(count * size) return_count_size(size_t count, size_t size);
// CHECK: |-FunctionDecl [[func_return_count_size:0x[^ ]+]] {{.+}} return_count_size
// CHECK: | |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK: | `-ParmVarDecl [[var_size:0x[^ ]+]]

void calls(void) {
  int arr[10];
  param_with_count(arr, 12);

  int *buf = return_count_size(10, 13);
}
// CHECK: `-FunctionDecl [[func_calls:0x[^ ]+]] {{.+}} calls
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | | |-BoundsCheckExpr {{.+}} 'void' 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && 12 - 2 <= __builtin_get_pointer_upper_bound(arr) - arr && 0 <= 12 - 2'
// CHECK:     | | | |-CallExpr
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(len - 2), int)' <FunctionToPointerDecay>
// CHECK:     | | | | | `-DeclRefExpr {{.+}} [[func_param_with_count]]
// CHECK:     | | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len - 2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     | | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK:     | | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK:     | | |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     | | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |     | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'long'
// CHECK:     | | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     | | |     |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK:     | | |     |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:     | | |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | | |       | `-IntegerLiteral {{.+}} 0
// CHECK:     | | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
// CHECK:     | | |-OpaqueValueExpr [[ove]]
// CHECK:     | | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | | |   `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:     | | |-OpaqueValueExpr [[ove_1]]
// CHECK:     | | | `-IntegerLiteral {{.+}} 12
// CHECK:     | | `-OpaqueValueExpr [[ove_2]]
// CHECK:     | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '-'
// CHECK:     | |       |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:     | |       `-IntegerLiteral {{.+}} 2
// CHECK:     | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:     | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_buf_1:0x[^ ]+]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:           `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:             |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:             | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:             | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'void *__single __sized_by(count * size)':'void *__single'
// CHECK:             | | |   |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:             | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:             | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:             | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:             | | |   |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__single __sized_by(count * size)':'void *__single'
// CHECK:             | | |   `-AssumptionExpr
// CHECK:             | | |     |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK:             | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:             | | |       | `-OpaqueValueExpr [[ove_6]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:             | | |         `-IntegerLiteral {{.+}} 0
// CHECK:             | |-OpaqueValueExpr [[ove_4]]
// CHECK:             | | `-ImplicitCastExpr {{.+}} 'size_t':'unsigned long' <IntegralCast>
// CHECK:             | |   `-IntegerLiteral {{.+}} 10
// CHECK:             | |-OpaqueValueExpr [[ove_5]]
// CHECK:             | | `-ImplicitCastExpr {{.+}} 'size_t':'unsigned long' <IntegralCast>
// CHECK:             | |   `-IntegerLiteral {{.+}} 13
// CHECK:             | |-OpaqueValueExpr [[ove_3]]
// CHECK:             | | `-CallExpr
// CHECK:             | |   |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(count * size)(*__single)(size_t, size_t)' <FunctionToPointerDecay>
// CHECK:             | |   | `-DeclRefExpr {{.+}} [[func_return_count_size]]
// CHECK:             | |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'size_t':'unsigned long'
// CHECK:             | `-OpaqueValueExpr [[ove_6]]
// CHECK:             |   `-BinaryOperator {{.+}} 'size_t':'unsigned long' '*'
// CHECK:             |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'size_t':'unsigned long'
// CHECK:             |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'size_t':'unsigned long'
// CHECK:             |-OpaqueValueExpr [[ove_4]] {{.*}} 'size_t':'unsigned long'
// CHECK:             |-OpaqueValueExpr [[ove_5]] {{.*}} 'size_t':'unsigned long'
// CHECK:             |-OpaqueValueExpr [[ove_3]] {{.*}} 'void *__single __sized_by(count * size)':'void *__single'
// CHECK:             `-OpaqueValueExpr [[ove_6]] {{.*}} 'size_t':'unsigned long'

