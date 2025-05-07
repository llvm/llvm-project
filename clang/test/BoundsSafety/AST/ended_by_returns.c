
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2> /dev/null | FileCheck %s

#include <ptrcheck.h>

// CHECK:      {{^}}|-FunctionDecl [[func_chunk:0x[^ ]+]] {{.+}} chunk
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_begin:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_end:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-BoundsCheckExpr {{.+}} 'end <= __builtin_get_pointer_upper_bound(begin + 1) && begin + 1 <= end && __builtin_get_pointer_lower_bound(begin + 1) <= begin + 1'
// CHECK-NEXT: {{^}}|       |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|       | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       | |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|       |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|       |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|       |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|       | `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|       |   |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|       |   | |-DeclRefExpr {{.+}} [[var_begin]]
// CHECK-NEXT: {{^}}|       |   | |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   | | `-DeclRefExpr {{.+}} [[var_end]]
// CHECK-NEXT: {{^}}|       |   | `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|       |   |   `-DeclRefExpr {{.+}} [[var_begin]]
// CHECK-NEXT: {{^}}|       |   `-IntegerLiteral {{.+}} 1
// CHECK-NEXT: {{^}}|       `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|         `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|           |-DeclRefExpr {{.+}} [[var_end]]
// CHECK-NEXT: {{^}}|           |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|           | `-DeclRefExpr {{.+}} [[var_end]]
// CHECK-NEXT: {{^}}|           `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|             `-DeclRefExpr {{.+}} [[var_begin]]
char *__ended_by(end) chunk(char *__ended_by(end) begin, char *end) {
    return begin + 1;
}

// CHECK: {{^}}|-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DeclStmt
// CHECK-NEXT: {{^}}|     `-VarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|           | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}|           | | | |-CallExpr
// CHECK-NEXT: {{^}}|           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|           | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}|           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|           | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|           | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|           | | |   | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|           | | |     |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|           | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|           | |     `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK-NEXT: {{^}}|           | `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|           |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|           |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|           |       | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK-NEXT: {{^}}|           |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|           |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
void foo(void) {
    int arr[10];
    int *p = chunk(arr, arr+10);
}

// CHECK:      {{^}}|-FunctionDecl [[func_fooCast:0x[^ ]+]] {{.+}} fooCast
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_arr_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DeclStmt
// CHECK-NEXT: {{^}}|     `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|       `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|           | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}|           | | | |-CallExpr
// CHECK-NEXT: {{^}}|           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|           | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}|           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|           | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|           | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|           | | |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|           | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|           | | |     |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|           | | |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           | |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|           | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|           | |     `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK-NEXT: {{^}}|           | `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|           |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|           |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|           |       | `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK-NEXT: {{^}}|           |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|           |-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|           `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
void fooCast(void) {
    int arr[10];
    int *p = (int*)chunk(arr, arr+10);
}

struct DataWithEndedBy {
    int *__ended_by(fend) fbegin;
    int *fend;
};

// CHECK: {{^}}|-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_data:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_arr_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(chunk(arr, arr + 10)) && chunk(arr, arr + 10) <= arr + 10 && __builtin_get_pointer_lower_bound(chunk(arr, arr + 10)) <= chunk(arr, arr + 10)'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | |   `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |     | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|   | |     | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}|   | |     | | | |-CallExpr
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     | | |   | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |     | | |     |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | |-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK-NEXT: {{^}}|   | |     | `-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK-NEXT: {{^}}|   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|   | |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   |     | `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->fend
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__bidi_indexable'
void bar(struct DataWithEndedBy *data) {
    int arr[10];
    data->fbegin = chunk(arr, arr+10);
    data->fend = arr+10;
}

// CHECK:      {{^}}|-FunctionDecl [[func_barCast:0x[^ ]+]] {{.+}} barCast
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_data_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_arr_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound((int *)chunk(arr, arr + 10)) && (int *)chunk(arr, arr + 10) <= arr + 10 && __builtin_get_pointer_lower_bound((int *)chunk(arr, arr + 10)) <= (int *)chunk(arr, arr + 10)'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|   | | `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|   | |     | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}|   | |     | | | |-CallExpr
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     | | |   | |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |   | `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |     | | |     |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | |-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK-NEXT: {{^}}|   | |     | `-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK-NEXT: {{^}}|   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|   | |     |-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|   |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   |     | `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK-NEXT: {{^}}|   |     `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->fend
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
void barCast(struct DataWithEndedBy *data) {
    int arr[10];
    data->fbegin = (int*)chunk(arr, arr+10);
    data->fend = arr+10;
}

struct DataWithCountedBy {
    long *__counted_by(count) ptr;
    unsigned long long count;
};

// CHECK: {{^}}|-FunctionDecl [[func_baz:0x[^ ]+]] {{.+}} baz
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_data_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_arr_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'chunk(arr, arr + 10) <= __builtin_get_pointer_upper_bound(chunk(arr, arr + 10)) && __builtin_get_pointer_lower_bound(chunk(arr, arr + 10)) <= chunk(arr, arr + 10) && len <= __builtin_get_pointer_upper_bound(chunk(arr, arr + 10)) - chunk(arr, arr + 10)'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'long *__single __counted_by(count)':'long *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'long *__single __counted_by(count)':'long *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | |         | | | |   `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     |-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:      {{^}}|   | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |         |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |         | `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |         `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |           `-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'long *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}|   | |     | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}|   | |     | | | |-CallExpr
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}|   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | | |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     | | |   | |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |   | `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |     | | |     |   `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     | | |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | | |-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     | |-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_4]]
// CHECK-NEXT: {{^}}|   | |     | `-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}|   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_4]]
// CHECK-NEXT: {{^}}|   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}|   | |     |-OpaqueValueExpr [[ove_15]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |       `-DeclRefExpr {{.+}} [[var_len]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->count
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_17]] {{.*}} 'unsigned long long'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_14]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_17]] {{.*}} 'unsigned long long'
void baz(struct DataWithCountedBy *data, int len) {
    int arr[10];
    data->ptr = chunk(arr, arr+10);
    data->count = len;
}

// CHECK:      {{^}}`-FunctionDecl [[func_bazCast:0x[^ ]+]] {{.+}} bazCast
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_data_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_arr_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |-BoundsCheckExpr {{.+}} '(long *)chunk(arr, arr + 10) <= __builtin_get_pointer_upper_bound((long *)chunk(arr, arr + 10)) && __builtin_get_pointer_lower_bound((long *)chunk(arr, arr + 10)) <= (long *)chunk(arr, arr + 10) && len <= __builtin_get_pointer_upper_bound((long *)chunk(arr, arr + 10)) - (long *)chunk(arr, arr + 10)'
// CHECK-NEXT: {{^}}    | | |-BinaryOperator {{.+}} 'long *__single __counted_by(count)':'long *__single' '='
// CHECK-NEXT: {{^}}    | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}    | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | | |   `-DeclRefExpr {{.+}} [[var_data_3]]
// CHECK-NEXT: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'long *__single __counted_by(count)':'long *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | |   `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | | |         | | | | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | | |         | | | |   `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |   | | |-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |   | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    | |   | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}    | |   |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |   |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |     |-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:      {{^}}    | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK-NEXT: {{^}}    | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    | |         |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    | |         | `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |         `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |           `-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}    | | `-CStyleCastExpr {{.+}} 'long *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK-NEXT: {{^}}    | |     | | |-BoundsCheckExpr {{.+}} 'arr + 10 <= __builtin_get_pointer_upper_bound(arr) && arr <= arr + 10 && __builtin_get_pointer_lower_bound(arr) <= arr'
// CHECK-NEXT: {{^}}    | |     | | | |-CallExpr
// CHECK-NEXT: {{^}}    | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK-NEXT: {{^}}    | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | | | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | | |   `-OpaqueValueExpr [[ove_20]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |     | | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    | |     | | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |     | | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |   | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    | |     | | |   | |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |     | | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |   |   | `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |   |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |     | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}    | |     | | |     |   `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | |     | | |       `-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | | |-OpaqueValueExpr [[ove_20]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     | |-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}    | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}    | |     | |     `-DeclRefExpr {{.+}} [[var_arr_5]]
// CHECK-NEXT: {{^}}    | |     | `-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}    | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}    | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK-NEXT: {{^}}    | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}    | |     |       | `-DeclRefExpr {{.+}} [[var_arr_5]]
// CHECK-NEXT: {{^}}    | |     |       `-IntegerLiteral {{.+}} 10
// CHECK-NEXT: {{^}}    | |     |-OpaqueValueExpr [[ove_19]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'char *__bidi_indexable'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_21]]
// CHECK-NEXT: {{^}}    |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK-NEXT: {{^}}    |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |       `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK-NEXT: {{^}}    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}      |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK-NEXT: {{^}}      | |-MemberExpr {{.+}} ->count
// CHECK-NEXT: {{^}}      | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}      | |   `-DeclRefExpr {{.+}} [[var_data_3]]
// CHECK-NEXT: {{^}}      | `-OpaqueValueExpr [[ove_21]] {{.*}} 'unsigned long long'
// CHECK:      {{^}}      |-OpaqueValueExpr [[ove_18]] {{.*}} 'long *__bidi_indexable'
// CHECK:      {{^}}      `-OpaqueValueExpr [[ove_21]] {{.*}} 'unsigned long long'
void bazCast(struct DataWithCountedBy *data, int len) {
    int arr[10];
    data->ptr = (long*)chunk(arr, arr+10);
    data->count = len;
}
