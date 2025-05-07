// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -Wcast-qual 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s -Wcast-qual 2>&1 | FileCheck %s

#include <ptrcheck.h>

void foo(void) {
    char *dst, *src;
    __builtin_memcpy(dst, src, 10);
}

// CHECK: {{^}}|-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | |-VarDecl [[var_dst:0x[^ ]+]]
// CHECK: {{^}}|   | `-VarDecl [[var_src:0x[^ ]+]]
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}|     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: {{^}}|     | | |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     | | |   | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     | | |   | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}|     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}|     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}|     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
// CHECK: {{^}}|     | | |   `-AssumptionExpr
// CHECK: {{^}}|     | | |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK: {{^}}|     | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | | |       | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | | |         `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|     | |     `-DeclRefExpr {{.+}} [[var_dst]]
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'const void *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     | |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|     | |     `-DeclRefExpr {{.+}} [[var_src]]
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}|     | |   `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}|     |   `-BoundsCheckExpr
// CHECK: {{^}}|     |     |-BoundsCheckExpr
// CHECK: {{^}}|     |     | |-CallExpr
// CHECK: {{^}}|     |     | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)(*)(void *__single __sized_by(function-parameter-0-2), const void *__single __sized_by(function-parameter-0-2), unsigned long)' <BuiltinFnToFnPtr>
// CHECK: {{^}}|     |     | | | `-DeclRefExpr {{.+}}
// CHECK: {{^}}|     |     | | |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     | | |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(function-parameter-0-2)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |     | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     |     |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     |     |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |     |   | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |   | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     |   | | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     |     |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |     |   |   |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     |     |   |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     |   |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |     |     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     |     |     `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}|     |     |       `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     |     |         |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |         | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     |     |         |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     |     |         |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     |         `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |     |           `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     |     |             `-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |       | | |-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |       | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |       | | `-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |       | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     |       | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |       |   |-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |       |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     |       |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |       |   `-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |       |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     |         |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     |         `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}|     |           `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     |             |-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |             | `-CStyleCastExpr {{.+}} 'const char *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     |             |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     |             |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |             `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     |               `-CStyleCastExpr {{.+}} 'const char *__bidi_indexable' <BitCast>
// CHECK: {{^}}|     |                 `-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_1]] {{.*}} 'void *__bidi_indexable'
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_2]] {{.*}} 'const void *__bidi_indexable'
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_3]] {{.*}} 'unsigned long'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove]] {{.*}} 'void *__single __sized_by(function-parameter-0-2)':'void *__single'
