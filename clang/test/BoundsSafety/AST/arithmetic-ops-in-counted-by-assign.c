
// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>
#include <stddef.h>

void param_with_count(int *__counted_by(len - 2) buf, int len) {
  int arr[10];
  len = 12;
  buf = arr;
}
// CHECK: -FunctionDecl [[func_param_with_count:0x[^ ]+]] {{.+}} param_with_count
// CHECK:  |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK:  |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK:  | `-DependerDeclsAttr
// CHECK:  `-CompoundStmt
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK:    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:    | |-BoundsCheckExpr {{.+}} 'int' 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && 12 - 2 <= __builtin_get_pointer_upper_bound(arr) - arr && 0 <= 12 - 2'
// CHECK:    | | |-BinaryOperator {{.+}} 'int' '='
// CHECK:    | | | |-DeclRefExpr {{.+}} [[var_len]]
// CHECK:    | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:    | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    | | | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    | | | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    | | | | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | | | | `-GetBoundExpr {{.+}} upper
// CHECK:    | | | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | | | |   |-GetBoundExpr {{.+}} lower
// CHECK:    | | | |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    | | | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | | |   | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'long'
// CHECK:    | | |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:    | | |   |   |-GetBoundExpr {{.+}} upper
// CHECK:    | | |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:    | | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:    | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | | |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:    | | |     | `-IntegerLiteral {{.+}} 0
// CHECK:    | | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
// CHECK:    | | `-OpaqueValueExpr [[ove_2]]
// CHECK:    | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:    | |     `-BinaryOperator {{.+}} 'int' '-'
// CHECK:    | |       |-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:    | |       `-IntegerLiteral {{.+}} 2
// CHECK:    | |-OpaqueValueExpr [[ove]]
// CHECK:    | | `-IntegerLiteral {{.+}} 12
// CHECK:    | `-OpaqueValueExpr [[ove_1]]
// CHECK:    |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:    |     `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:      |-BinaryOperator {{.+}} 'int *__single __counted_by(len - 2)':'int *__single' '='
// CHECK:      | |-DeclRefExpr {{.+}} [[var_buf]]
// CHECK:      | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len - 2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:      |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK:      |-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:      `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'

void local_count(void) {
  int arr[10];
  int len = 8;
  int *__counted_by(len + 2) buf = arr;
}
// CHECK: -FunctionDecl [[func_local_count:0x[^ ]+]] {{.+}} local_count
// CHECK:  `-CompoundStmt
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_arr_1:0x[^ ]+]]
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_len_1:0x[^ ]+]]
// CHECK:    |   |-IntegerLiteral {{.+}} 8
// CHECK:    |   `-DependerDeclsAttr
// CHECK:    `-DeclStmt
// CHECK:      `-VarDecl [[var_buf_1:0x[^ ]+]]
// CHECK:        `-BoundsCheckExpr {{.+}} 'int *__single __counted_by(len + 2)':'int *__single' 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && len + 2 <= __builtin_get_pointer_upper_bound(arr) - arr && 0 <= len + 2'
// CHECK:          |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len + 2)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:          | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:          |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:          | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:          | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          | | | `-GetBoundExpr {{.+}} upper
// CHECK:          | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:          | |   |-GetBoundExpr {{.+}} lower
// CHECK:          | |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:          | |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:          |   | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'long'
// CHECK:          |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:          |   |   |-GetBoundExpr {{.+}} upper
// CHECK:          |   |   | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:          |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK:          |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:          |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:          |     | `-IntegerLiteral {{.+}} 0
// CHECK:          |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'long'
// CHECK:          |-OpaqueValueExpr [[ove_3]]
// CHECK:          | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:          |   `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK:          `-OpaqueValueExpr [[ove_4]]
// CHECK:            `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:              `-BinaryOperator {{.+}} 'int' '+'
// CHECK:                |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                | `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:                `-IntegerLiteral {{.+}} 2

void local_count_size(void) {
  size_t nelems;
  size_t size;
  void *__sized_by(nelems * size) buf;
  int arr[10];
  nelems = 10;
  size = 4;
  buf = arr;
}
// CHECK: -FunctionDecl [[func_local_count_size:0x[^ ]+]] {{.+}} local_count_size
// CHECK:  `-CompoundStmt
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_nelems:0x[^ ]+]]
// CHECK:    |   `-DependerDeclsAttr
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_size:0x[^ ]+]]
// CHECK:    |   `-DependerDeclsAttr
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_buf_2:0x[^ ]+]]
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_arr_2:0x[^ ]+]]
// CHECK:    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:    | |-BoundsCheckExpr {{.+}} 'size_t':'unsigned long' 'arr <= __builtin_get_pointer_upper_bound(arr) && __builtin_get_pointer_lower_bound(arr) <= arr && 10 * 4 <= (char *)__builtin_get_pointer_upper_bound(arr) - (char *__bidi_indexable)arr'
// CHECK:    | | |-BinaryOperator {{.+}} 'size_t':'unsigned long' '='
// CHECK:    | | | |-DeclRefExpr {{.+}} [[var_nelems]]
// CHECK:    | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'size_t':'unsigned long'
// CHECK:    | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:    | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:    | |   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:    | |   | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:    | |   |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:    | |   |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:    | |     |-BinaryOperator {{.+}} 'size_t':'unsigned long' '*'
// CHECK:    | |     | |-OpaqueValueExpr [[ove_5]] {{.*}} 'size_t':'unsigned long'
// CHECK:    | |     | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'size_t':'unsigned long'
// CHECK:    | |     `-ImplicitCastExpr {{.+}} 'size_t':'unsigned long' <IntegralCast>
// CHECK:    | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:    | |         |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:    | |         | `-GetBoundExpr {{.+}} upper
// CHECK:    | |         |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:    | |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:    | |             `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:    | |-OpaqueValueExpr [[ove_5]]
// CHECK:    | | `-ImplicitCastExpr {{.+}} 'size_t':'unsigned long' <IntegralCast>
// CHECK:    | |   `-IntegerLiteral {{.+}} 10
// CHECK:    | |-OpaqueValueExpr [[ove_7]]
// CHECK:    | | `-ImplicitCastExpr {{.+}} 'size_t':'unsigned long' <IntegralCast>
// CHECK:    | |   `-IntegerLiteral {{.+}} 4
// CHECK:    | `-OpaqueValueExpr [[ove_6]]
// CHECK:    |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:    |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:    |       `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:    |-BinaryOperator {{.+}} 'size_t':'unsigned long' '='
// CHECK:    | |-DeclRefExpr {{.+}} [[var_size]]
// CHECK:    | `-OpaqueValueExpr [[ove_7]] {{.*}} 'size_t':'unsigned long'
// CHECK:    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:      |-BinaryOperator {{.+}} 'void *__single __sized_by(nelems * size)':'void *__single' '='
// CHECK:      | |-DeclRefExpr {{.+}} [[var_buf_2]]
// CHECK:      | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(nelems * size)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:      |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK:      |-OpaqueValueExpr [[ove_5]] {{.*}} 'size_t':'unsigned long'
// CHECK:      |-OpaqueValueExpr [[ove_7]] {{.*}} 'size_t':'unsigned long'
// CHECK:      `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
