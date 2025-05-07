

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

typedef struct {
  int len;
  int offs;
	int fam[__counted_by(len - offs)];
} S;

void f(S *s) {
  int arr[10] = {0};
  s = (S *)&arr[5];
  s->offs = 5;
  s->len = 10;
}
// CHECK: `-FunctionDecl [[func_f:0x[^ ]+]] {{.+}} f
// CHECK:   |-ParmVarDecl [[var_s:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK:     |   `-InitListExpr
// CHECK:     |     |-array_filler: ImplicitValueInitExpr
// CHECK:     |     `-IntegerLiteral {{.+}} 0
// CHECK:     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     | |-BinaryOperator {{.+}} 'S *__single' '='
// CHECK:     | | |-DeclRefExpr {{.+}} [[var_s]]
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'S *__single' <BoundsSafetyPointerCast>
// CHECK:     | |   `-PredefinedBoundsCheckExpr {{.+}} 'S *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:     | |     |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'S *__bidi_indexable'
// CHECK:     | |     |-OpaqueValueExpr [[ove]] {{.*}} 'S *__bidi_indexable'
// CHECK:     | |     |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK:     | |     | `-MemberExpr {{.+}} ->fam
// CHECK:     | |     |   `-OpaqueValueExpr [[ove]] {{.*}} 'S *__bidi_indexable'
// CHECK:     | |     `-BinaryOperator {{.+}} 'int' '-'
// CHECK:     | |       |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |       `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:     | |-OpaqueValueExpr [[ove]]
// CHECK:     | | `-CStyleCastExpr {{.+}} 'S *__bidi_indexable' <BitCast>
// CHECK:     | |   `-UnaryOperator {{.+}} cannot overflow
// CHECK:     | |     `-ArraySubscriptExpr
// CHECK:     | |       |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK:     | |       | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK:     | |       `-IntegerLiteral {{.+}} 5
// CHECK:     | |-OpaqueValueExpr [[ove_2]]
// CHECK:     | | `-IntegerLiteral {{.+}} 5
// CHECK:     | `-OpaqueValueExpr [[ove_1]]
// CHECK:     |   `-IntegerLiteral {{.+}} 10
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-MemberExpr {{.+}} ->offs
// CHECK:     | | `-ImplicitCastExpr {{.+}} 'S *__single' <LValueToRValue>
// CHECK:     | |   `-DeclRefExpr {{.+}} [[var_s]]
// CHECK:     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:       |-BinaryOperator {{.+}} 'int' '='
// CHECK:       | |-MemberExpr {{.+}} ->len
// CHECK:       | | `-ImplicitCastExpr {{.+}} 'S *__single' <LValueToRValue>
// CHECK:       | |   `-DeclRefExpr {{.+}} [[var_s]]
// CHECK:       | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove]] {{.*}} 'S *__bidi_indexable'
// CHECK:       |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:       `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
