// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2> /dev/null | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2> /dev/null | FileCheck %s

#include <ptrcheck.h>

char *__ended_by(end) chunk(char *__ended_by(end) begin, char *end) {
    return begin + 1;
}
//CHECK: FunctionDecl [[func_chunk:0x[^ ]+]] {{.*}} used chunk 'char *__single __ended_by(end)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )'
//CHECK: |-ParmVarDecl [[var_begin:0x[^ ]+]] {{.*}} 'char *__single __ended_by(end)':'char *__single'
//CHECK: |-ParmVarDecl [[var_end:0x[^ ]+]] {{.*}} 'char *__single /* __started_by(begin) */ ':'char *__single'
//CHECK: `-CompoundStmt
//CHECK:   `-ReturnStmt
//CHECK:     `-ImplicitCastExpr {{.*}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
//CHECK:       `-BinaryOperator {{.*}} 'char *__bidi_indexable' '+'
//CHECK:         |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
//CHECK:         | |-DeclRefExpr {{.+}} [[var_begin]]
//CHECK:         | |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <LValueToRValue>
//CHECK:         | | `-DeclRefExpr {{.+}} [[var_end]]
//CHECK:         | `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <LValueToRValue>
//CHECK:         |   `-DeclRefExpr {{.+}} [[var_begin]]
//CHECK:         `-IntegerLiteral {{.*}} 'int' 1

// CHECK-LABEL: foo
void foo(void) {
    int arr[10];
    int *p = chunk(arr, arr+10);
}
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_p:0x[^ ]+]]
// CHECK:       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:           | | |-BoundsCheckExpr
// CHECK:           | | | |-CallExpr
// CHECK:           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:           | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:           | | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:           | | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |   | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |     | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | |-OpaqueValueExpr [[ove]]
// CHECK:           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:           | |     `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:           | `-OpaqueValueExpr [[ove_1]]
// CHECK:           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:           |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:           |       | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:           |       `-IntegerLiteral {{.+}} 10
// CHECK:           |-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK:           `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'

// CHECK-LABEL: fooCast
void fooCast(void) {
    int arr[10];
    int *p = (int*)chunk(arr, arr+10);
}
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_1:0x[^ ]+]]
// CHECK:   `-DeclStmt
// CHECK:     `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK:       `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:           | | |-BoundsCheckExpr
// CHECK:           | | | |-CallExpr
// CHECK:           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:           | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:           | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:           | | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:           | | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |   | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           | | |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'
// CHECK:           | |-OpaqueValueExpr [[ove_2]]
// CHECK:           | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:           | |     `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK:           | `-OpaqueValueExpr [[ove_3]]
// CHECK:           |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:           |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:           |       | `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK:           |       `-IntegerLiteral {{.+}} 10
// CHECK:           |-OpaqueValueExpr [[ove_2]] {{.*}} 'char *__bidi_indexable'
// CHECK:           `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__bidi_indexable'

struct DataWithEndedBy {
    int *__ended_by(fend) fbegin;
    int *fend;
};

// CHECK-LABEL: bar
void bar(struct DataWithEndedBy *data) {
    int arr[10];
    data->fbegin = chunk(arr, arr+10);
    data->fend = arr+10;
}
// CHECK: |-ParmVarDecl [[var_data:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_2:0x[^ ]+]]
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK:   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK:   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:   | | | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK:   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:   | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |         | | | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | |         | | | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-GetBoundExpr {{.+}} lower
// CHECK:   | |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_4]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-BoundsCheckExpr
// CHECK:   | |     | | | |-CallExpr
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:   | |     | | |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |     | `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | |-OpaqueValueExpr [[ove_5]]
// CHECK:   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:   | |     | `-OpaqueValueExpr [[ove_6]]
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK:   | |     |-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | `-OpaqueValueExpr [[ove_7]]
// CHECK:   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |     | `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:   |     `-IntegerLiteral {{.+}} 10
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK:     | |-MemberExpr {{.+}} ->fend
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'

// CHECK-LABEL: barCast
void barCast(struct DataWithEndedBy *data) {
    int arr[10];
    data->fbegin = (int*)chunk(arr, arr+10);
    data->fend = arr+10;
}
// CHECK: |-ParmVarDecl [[var_data_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_3:0x[^ ]+]]
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK:   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK:   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:   | | | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK:   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:   | | |   `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | |         | | | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | |         | | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-GetBoundExpr {{.+}} lower
// CHECK:   | |     | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_8]]
// CHECK:   | | `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-BoundsCheckExpr
// CHECK:   | |     | | | |-CallExpr
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:   | |     | | |   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |     | `-OpaqueValueExpr [[ove_9]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |       `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | |-OpaqueValueExpr [[ove_9]]
// CHECK:   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK:   | |     | `-OpaqueValueExpr [[ove_10]]
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK:   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK:   | |     |-OpaqueValueExpr [[ove_9]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | `-OpaqueValueExpr [[ove_11]]
// CHECK:   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |     | `-DeclRefExpr {{.+}} [[var_arr_3]]
// CHECK:   |     `-IntegerLiteral {{.+}} 10
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK:     | |-MemberExpr {{.+}} ->fend
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__bidi_indexable'

struct DataWithCountedBy {
    long *__counted_by(count) ptr;
    unsigned long long count;
};

// CHECK-LABEL: baz 'void (struct DataWithCountedBy *__single, int)'
void baz(struct DataWithCountedBy *data, int len) {
    int arr[10];
    data->ptr = chunk(arr, arr+10);
    data->count = len;
}
// CHECK: |-ParmVarDecl [[var_data_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_4:0x[^ ]+]]
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'long *__single __counted_by(count)':'long *__single' '='
// CHECK:   | | | |-MemberExpr {{.+}} ->ptr
// CHECK:   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK:   | | | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK:   | | | `-ImplicitCastExpr {{.+}} 'long *__single __counted_by(count)':'long *__single' <BoundsSafetyPointerCast>
// CHECK:   | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | | |         | | | | | `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | |         | | | |   `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | |         |-GetBoundExpr {{.+}} upper
// CHECK:   | |         | `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |         `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |           `-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_12]]
// CHECK:   | | `-ImplicitCastExpr {{.+}} 'long *__bidi_indexable' <BitCast>
// CHECK:   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-BoundsCheckExpr
// CHECK:   | |     | | | |-CallExpr
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:   | |     | | |   |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |     | `-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |       `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | |-OpaqueValueExpr [[ove_13]]
// CHECK:   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_4]]
// CHECK:   | |     | `-OpaqueValueExpr [[ove_14]]
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_4]]
// CHECK:   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK:   | |     |-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | `-OpaqueValueExpr [[ove_15]]
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |       `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK:     | |-MemberExpr {{.+}} ->count
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK:     | `-OpaqueValueExpr [[ove_15]] {{.*}} 'unsigned long long'
// CHECK:     |-OpaqueValueExpr [[ove_12]] {{.*}} 'long *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_15]] {{.*}} 'unsigned long long'

// CHECK-LABEL: bazCast 'void (struct DataWithCountedBy *__single, int)'
void bazCast(struct DataWithCountedBy *data, int len) {
    int arr[10];
    data->ptr = (long*)chunk(arr, arr+10);
    data->count = len;
}
// CHECK: |-ParmVarDecl [[var_data_3:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_5:0x[^ ]+]]
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'long *__single __counted_by(count)':'long *__single' '='
// CHECK:   | | | |-MemberExpr {{.+}} ->ptr
// CHECK:   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK:   | | | |   `-DeclRefExpr {{.+}} [[var_data_3]]
// CHECK:   | | | `-ImplicitCastExpr {{.+}} 'long *__single __counted_by(count)':'long *__single' <BoundsSafetyPointerCast>
// CHECK:   | | |   `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | | |         | | | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | |         | | | |   `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   | |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK:   | |         |-GetBoundExpr {{.+}} upper
// CHECK:   | |         | `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |         `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK:   | |           `-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_16]]
// CHECK:   | | `-CStyleCastExpr {{.+}} 'long *__bidi_indexable' <BitCast>
// CHECK:   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-BoundsCheckExpr
// CHECK:   | |     | | | |-CallExpr
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)(*__single)(char *__single __ended_by(end), char *__single /* __started_by(begin) */ )' <FunctionToPointerDecay>
// CHECK:   | |     | | | | | `-DeclRefExpr {{.+}} [[func_chunk]]
// CHECK:   | |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __ended_by(end)':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | | | `-OpaqueValueExpr [[ove_17]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | | `-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(begin) */ ':'char *__single' <BoundsSafetyPointerCast>
// CHECK:   | |     | | | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |     | | |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |   | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |   |   `-GetBoundExpr {{.+}} upper
// CHECK:   | |     | | |   |     `-OpaqueValueExpr [[ove_17]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     | | |     |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |     | `-OpaqueValueExpr [[ove_17]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |     `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:   | |     | | |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | | |-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     | |-OpaqueValueExpr [[ove_17]]
// CHECK:   | |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     | |     `-DeclRefExpr {{.+}} [[var_arr_5]]
// CHECK:   | |     | `-OpaqueValueExpr [[ove_18]]
// CHECK:   | |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:   | |     |     `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:   | |     |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     |       | `-DeclRefExpr {{.+}} [[var_arr_5]]
// CHECK:   | |     |       `-IntegerLiteral {{.+}} 10
// CHECK:   | |     |-OpaqueValueExpr [[ove_17]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK:   | `-OpaqueValueExpr [[ove_19]]
// CHECK:   |   `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK:   |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:   |       `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'unsigned long long' '='
// CHECK:     | |-MemberExpr {{.+}} ->count
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct DataWithCountedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_data_3]]
// CHECK:     | `-OpaqueValueExpr [[ove_19]] {{.*}} 'unsigned long long'
// CHECK:     |-OpaqueValueExpr [[ove_16]] {{.*}} 'long *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_19]] {{.*}} 'unsigned long long'
