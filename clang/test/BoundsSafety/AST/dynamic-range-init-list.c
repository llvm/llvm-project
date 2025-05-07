

// RUN: %clang_cc1 -fbounds-safety -ast-dump -triple x86_64 %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -triple x86_64 %s | FileCheck %s

#include <ptrcheck.h>

struct RangePtrs {
  int *__ended_by(iter) start;
  int *__ended_by(end) iter;
  void *end;
};

void Test(void) {
  int arr[10];
  struct RangePtrs rptrs = { arr + 1, arr + 2, arr + 3 };
}

// CHECK: {{^}}`-FunctionDecl [[func_Test:0x[^ ]+]] {{.+}} Test
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_arr:0x[^ ]+]]
// CHECK: {{^}}    `-DeclStmt
// CHECK: {{^}}      `-VarDecl [[var_rptrs:0x[^ ]+]]
// CHECK: {{^}}        `-BoundsCheckExpr {{.+}} 'arr + 2 <= __builtin_get_pointer_upper_bound(arr + 1) && arr + 1 <= arr + 2 && __builtin_get_pointer_lower_bound(arr + 1) <= arr + 1'
// CHECK: {{^}}          |-BoundsCheckExpr {{.+}} 'arr + 3 <= __builtin_get_pointer_upper_bound(arr + 2) && arr + 2 <= arr + 3 && __builtin_get_pointer_lower_bound(arr + 2) <= arr + 2'
// CHECK: {{^}}          | |-InitListExpr
// CHECK: {{^}}          | | |-ImplicitCastExpr {{.+}} 'int *__single __ended_by(iter)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | | |-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end) /* __started_by(start) */ ':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | | `-ImplicitCastExpr {{.+}} 'void *__single /* __started_by(iter) */ ':'void *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}          | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}          |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}          |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          |   | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}          |   | | `-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}          |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}          |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          |   |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |   |   `-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}          |   |     `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          |   |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}          |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          |     |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}          |     | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}          | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}          | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}          | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          | |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}          |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}          |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}          |-OpaqueValueExpr [[ove]]
// CHECK: {{^}}          | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}          |   |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: {{^}}          |   | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: {{^}}          |   `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}          |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}          | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}          |   |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: {{^}}          |   | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: {{^}}          |   `-IntegerLiteral {{.+}} 2
// CHECK: {{^}}          `-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}            `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK: {{^}}              `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}                |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <ArrayToPointerDecay>
// CHECK: {{^}}                | `-DeclRefExpr {{.+}} [[var_arr]]
// CHECK: {{^}}                `-IntegerLiteral {{.+}} 3

