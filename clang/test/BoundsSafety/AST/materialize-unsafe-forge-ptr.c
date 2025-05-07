
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s

#include <ptrcheck.h>

void *g;

void test() {
  const void *__sized_by(4) ptr = __unsafe_forge_bidi_indexable(const void *, g, 4);
}

// CHECK: TranslationUnitDecl
// CHECK: |-VarDecl [[var_g:0x[^ ]+]]
// CHECK: `-FunctionDecl [[func_test:0x[^ ]+]] {{.+}} test
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_ptr:0x[^ ]+]]
// CHECK:         `-BoundsCheckExpr
// CHECK:           |-ImplicitCastExpr {{.+}} 'const void *__single __sized_by(4)':'const void *__single' <BoundsSafetyPointerCast>
// CHECK:           | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK:           | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           | | | `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | |   |-GetBoundExpr {{.+}} lower
// CHECK:           | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'const void *' <BoundsSafetyPointerCast>
// CHECK:           | |     `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |   | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'long'
// CHECK:           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:           |   |   |-CStyleCastExpr {{.+}} 'const char *' <BitCast>
// CHECK:           |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:           |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           |   |   `-ImplicitCastExpr {{.+}} 'const char *' <BoundsSafetyPointerCast>
// CHECK:           |   |     `-CStyleCastExpr {{.+}} 'const char *__bidi_indexable' <BitCast>
// CHECK:           |   |       `-OpaqueValueExpr [[ove]] {{.*}} 'const void *__bidi_indexable'
// CHECK:           |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:           |     | `-IntegerLiteral {{.+}} 0
// CHECK:           |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'long'
// CHECK:           |-OpaqueValueExpr [[ove]]
// CHECK:           | `-ParenExpr
// CHECK:           |   `-CStyleCastExpr {{.+}} 'const void *__bidi_indexable' <NoOp>
// CHECK:           |     `-ForgePtrExpr
// CHECK:           |       |-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK:           |       | `-ParenExpr
// CHECK:           |       |   `-DeclRefExpr {{.+}} [[var_g]]
// CHECK:           |       `-ParenExpr
// CHECK:           |         `-IntegerLiteral {{.+}} 4
// CHECK:           `-OpaqueValueExpr [[ove_1]]
// CHECK:             `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:               `-IntegerLiteral {{.+}} 4
