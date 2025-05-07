// RUN: %clang_cc1 -ast-dump -fbounds-safety -isystem %S/Inputs/system-merge-dynamic-bound-attr %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -isystem %S/Inputs/system-merge-dynamic-bound-attr %s 2>&1 | FileCheck %s

#include <header-with-attr.h>
#include <header-no-attr.h>

// CHECK: {{^}}|-FunctionDecl {{.+}} myalloc
// CHECK: {{^}}| |-ParmVarDecl
// CHECK: {{^}}| `-AllocSizeAttr
// CHECK: {{^}}|-FunctionDecl {{.+}} myalloc
// CHECK: {{^}}| |-ParmVarDecl
// CHECK: {{^}}| `-AllocSizeAttr
// CHECK: {{^}}|-FunctionDecl {{.+}} memcpy
// CHECK: {{^}}| |-ParmVarDecl
// CHECK: {{^}}| |-ParmVarDecl
// CHECK: {{^}}| `-ParmVarDecl
// CHECK: {{^}}|   `-DependerDeclsAttr

void Test(unsigned siz) {
// CHECK: {{^}}`-FunctionDecl [[func_Test:0x[^ ]+]] {{.+}} Test
// CHECK: {{^}}  |-ParmVarDecl [[var_siz:0x[^ ]+]]

  void *src = myalloc(siz);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_src:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'void *(*__single)(unsigned int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} 'myalloc'
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove]] {{.*}} 'void *'

  void *dst = myalloc(siz);
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_dst:0x[^ ]+]]
// CHECK: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}    |     | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}    |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    |     | | |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'
// CHECK: {{^}}    |     | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     | |-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}    |     | `-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}    |     |   `-CallExpr
// CHECK: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'void *(*__single)(unsigned int)' <FunctionToPointerDecay>
// CHECK: {{^}}    |     |     | `-DeclRefExpr {{.+}} 'myalloc'
// CHECK: {{^}}    |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned int'
// CHECK: {{^}}    |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'void *'

  memcpy(dst, src, siz);
// CHECK: {{^}}    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}      |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}      | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}      | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: {{^}}      | | |   | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      | | |   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      | | |   | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}      | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}      | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}      | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: {{^}}      | | |   `-AssumptionExpr
// CHECK: {{^}}      | | |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: {{^}}      | | |       |-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: {{^}}      | | |       | `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      | | |       `-ImplicitCastExpr {{.+}} 'long long' <IntegralCast>
// CHECK: {{^}}      | | |         `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}      | |-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}      | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}      | |   `-DeclRefExpr {{.+}} [[var_dst]]
// CHECK: {{^}}      | |-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}      | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}      | |   `-DeclRefExpr {{.+}} [[var_src]]
// CHECK: {{^}}      | |-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}      | | `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: {{^}}      | |   `-ImplicitCastExpr {{.+}} 'unsigned int' <LValueToRValue>
// CHECK: {{^}}      | |     `-DeclRefExpr {{.+}} [[var_siz]]
// CHECK: {{^}}      | `-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}      |   `-BoundsCheckExpr
// CHECK: {{^}}      |     |-BoundsCheckExpr
// CHECK: {{^}}      |     | |-CallExpr
// CHECK: {{^}}      |     | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*__single)(void *__single __sized_by(function-parameter-0-2), void *__single __sized_by(function-parameter-0-2), unsigned long long)' <FunctionToPointerDecay>
// CHECK: {{^}}      |     | | | `-DeclRefExpr {{.+}}
// CHECK: {{^}}      |     | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}      |     |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}      |     |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |     |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |   | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     |   | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}      |     |   | |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |     |   |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}      |     |   |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |     |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      |     |     `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: {{^}}      |     |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}      |     |         |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |         | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}      |     |         |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}      |     |         |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |     |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}      |     |             `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}      |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}      |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |       | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |       | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |       | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |       | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}      |       | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |       |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |       |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}      |       |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |       |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |       |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}      |         |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      |         `-ImplicitCastExpr {{.+}} 'unsigned long long' <IntegralCast>
// CHECK: {{^}}      |           `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}      |             |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |             | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}      |             |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}      |             |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |             `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}      |               `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}      |                 `-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |-OpaqueValueExpr [[ove_6]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}      |-OpaqueValueExpr [[ove_7]] {{.*}} 'unsigned long long'
// CHECK: {{^}}      `-OpaqueValueExpr [[ove_4]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
}
