
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2> /dev/null | FileCheck %s

#include <ptrcheck.h>

struct DataWithEndedBy {
  int *__ended_by(fend) fbegin;
  int *fend;
};

// CHECK-LABEL: test
void test(struct DataWithEndedBy *data, int len) {
  int arr[10];
  data->fbegin = arr;
  data->fend = arr + 10;
}
// CHECK: | |-ParmVarDecl [[var_data:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr
// CHECK: |   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK: |   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK: |   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK: |   | | | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     |-GetBoundExpr {{.+}} lower
// CHECK: |   | |     | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |       `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |-OpaqueValueExpr [[ove]]
// CHECK: |   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   | |   `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: |   | `-OpaqueValueExpr [[ove_1]]
// CHECK: |   |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   |     | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: |   |     `-IntegerLiteral {{.+}} 10
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK: |     | |-MemberExpr {{.+}} ->fend
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_data]]
// CHECK: |     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     |-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'

// CHECK-LABEL: test_bitcast
void test_bitcast(struct DataWithEndedBy *data, int len) {
  char arr[10];
  data->fbegin = arr;
  data->fend = arr + 10;
}
// CHECK: | |-ParmVarDecl [[var_data_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_arr_1:0x[^ ]+]]
// CHECK: |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   | |-BoundsCheckExpr
// CHECK: |   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK: |   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK: |   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK: |   | | | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK: |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |   | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: |   | |   | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   | |     |-GetBoundExpr {{.+}} lower
// CHECK: |   | |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: |   | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: |   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |   | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   | |     `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK: |   | `-OpaqueValueExpr [[ove_3]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |   |     `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK: |   |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: |   |       | `-DeclRefExpr {{.+}} [[var_arr_1]]
// CHECK: |   |       `-IntegerLiteral {{.+}} 10
// CHECK: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK: |     | |-MemberExpr {{.+}} ->fend
// CHECK: |     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK: |     | |   `-DeclRefExpr {{.+}} [[var_data_1]]
// CHECK: |     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK: |     |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__bidi_indexable'
// CHECK: |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__bidi_indexable'

// CHECK-LABEL: test_ext_bitcast
void test_ext_bitcast(struct DataWithEndedBy *data, int len) {
  char arr[10];
  data->fbegin = (int *)arr;
  data->fend = (int *)(arr + 10);
}
// CHECK: |-ParmVarDecl [[var_data_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   |-DeclStmt
// CHECK:   | `-VarDecl [[var_arr_2:0x[^ ]+]]
// CHECK:   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:   | |-BoundsCheckExpr
// CHECK:   | | |-BinaryOperator {{.+}} 'int *__single __ended_by(fend)':'int *__single' '='
// CHECK:   | | | |-MemberExpr {{.+}} ->fbegin
// CHECK:   | | | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:   | | | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK:   | | | `-ImplicitCastExpr {{.+}} 'int *__single __ended_by(fend)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:   | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK:   | |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:   | |     |-GetBoundExpr {{.+}} lower
// CHECK:   | |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:   | |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:   | |-OpaqueValueExpr [[ove_4]]
// CHECK:   | | `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:   | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   | |     `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:   | `-OpaqueValueExpr [[ove_5]]
// CHECK:   |   `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK:   |     `-ParenExpr
// CHECK:   |       `-BinaryOperator {{.+}} 'char *__bidi_indexable' '+'
// CHECK:   |         |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:   |         | `-DeclRefExpr {{.+}} [[var_arr_2]]
// CHECK:   |         `-IntegerLiteral {{.+}} 10
// CHECK:   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |-BinaryOperator {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' '='
// CHECK:     | |-MemberExpr {{.+}} ->fend
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'struct DataWithEndedBy *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_data_2]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single /* __started_by(fbegin) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK:     |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
// CHECK:     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__bidi_indexable'
