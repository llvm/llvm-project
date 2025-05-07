

// RUN: %clang_cc1 -fbounds-safety -ast-dump -Wno-bounds-safety-init-list %s | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -Wno-bounds-safety-init-list %s | FileCheck %s
#include <ptrcheck.h>

void Test(void) {
  int *__single iptr;
  int len;
  char *__counted_by(len) cptr = iptr;
  int len2;
  long *__counted_by(len2) lptr = cptr;
  return;
}

// CHECK-LABEL: Test
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_iptr:0x[^ ]+]]
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_len:0x[^ ]+]]
// CHECK: {{^}}    |   `-DependerDeclsAttr
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_cptr:0x[^ ]+]]
// CHECK: {{^}}    |   `-BoundsCheckExpr
// CHECK: {{^}}    |     |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     | | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     | |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}    |     | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     |   | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'long'
// CHECK: {{^}}    |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}    |     |   |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    |     |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    |     |     | `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'long'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |     | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}    |     |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: {{^}}    |     |       `-DeclRefExpr {{.+}} [[var_iptr]]
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}    |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |           `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_len2:0x[^ ]+]]
// CHECK: {{^}}    |   `-DependerDeclsAttr
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_lptr:0x[^ ]+]]
// CHECK: {{^}}    |   `-BoundsCheckExpr
// CHECK: {{^}}    |     |-ImplicitCastExpr {{.+}} 'long *__single __counted_by(len2)':'long *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     |       | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: {{^}}    |     |       | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     | | | |-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     | | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     | |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}    |     | |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     | |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     |   | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'long'
// CHECK: {{^}}    |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}    |     |   |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    |     |   |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     |   |   `-ImplicitCastExpr {{.+}} 'long *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'long *__bidi_indexable'
// CHECK: {{^}}    |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    |     |     | `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'long'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}    |     | `-ImplicitCastExpr {{.+}} 'long *__bidi_indexable' <BitCast>
// CHECK: {{^}}    |     |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: {{^}}    |     |     | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: {{^}}    |     |     | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    |     |     | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: {{^}}    |     |     | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK: {{^}}    |     |     | |-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}    |     |     | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <LValueToRValue>
// CHECK: {{^}}    |     |     | |   `-DeclRefExpr {{.+}} [[var_cptr]]
// CHECK: {{^}}    |     |     | `-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}    |     |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     |     |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: {{^}}    |     |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}    |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    |         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |           `-DeclRefExpr {{.+}} [[var_len2]]
