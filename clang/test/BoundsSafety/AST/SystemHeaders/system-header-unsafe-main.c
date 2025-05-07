
#include <system-header-unsafe-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix RELAXED
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix RELAXED
// expected-no-diagnostics

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix STRICT
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix STRICT

void func(char * __unsafe_indexable ptr, char * __bidi_indexable bidi) {
  funcInSDK(ptr, bidi);
}

// RELAXED: TranslationUnitDecl
// RELAXED: |-FunctionDecl [[func_funcWithAnnotation:0x[^ ]+]] {{.+}} funcWithAnnotation
// RELAXED: | |-ParmVarDecl [[var_foo:0x[^ ]+]]
// RELAXED: | `-ParmVarDecl [[var_bar:0x[^ ]+]]
//
// STRICT: TranslationUnitDecl
// STRICT: |-FunctionDecl [[func_funcWithAnnotation:0x[^ ]+]] {{.+}} funcWithAnnotation
// STRICT: | |-ParmVarDecl [[var_foo:0x[^ ]+]]
// STRICT: | `-ParmVarDecl [[var_bar:0x[^ ]+]]

// RELAXED: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// RELAXED: | |-ParmVarDecl [[var_ptr:0x[^ ]+]]
// RELAXED: | |-ParmVarDecl [[var_bidi:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |     |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |     | |-BoundsCheckExpr
// RELAXED: |     | | |-CallExpr
// RELAXED: |     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(char *__single __sized_by(4), char *__single __sized_by(5))' <FunctionToPointerDecay>
// RELAXED: |     | | | | `-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// RELAXED: |     | | | |-ImplicitCastExpr {{.+}} 'char *__single __sized_by(4)':'char *__single' <BoundsSafetyPointerCast>
// RELAXED: |     | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *'
// RELAXED: |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __sized_by(5)':'char *__single' <BoundsSafetyPointerCast>
// RELAXED: |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | | `-BinaryOperator {{.+}} 'int' '&&'
// RELAXED: |     | |   |-BinaryOperator {{.+}} 'int' '&&'
// RELAXED: |     | |   | |-BinaryOperator {{.+}} 'int' '<='
// RELAXED: |     | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |   | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |   | |   `-GetBoundExpr {{.+}} upper
// RELAXED: |     | |   | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |   | `-BinaryOperator {{.+}} 'int' '<='
// RELAXED: |     | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |   |   | `-GetBoundExpr {{.+}} lower
// RELAXED: |     | |   |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |   `-BinaryOperator {{.+}} 'int' '&&'
// RELAXED: |     | |     |-BinaryOperator {{.+}} 'int' '<='
// RELAXED: |     | |     | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'long'
// RELAXED: |     | |     | `-BinaryOperator {{.+}} 'long' '-'
// RELAXED: |     | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// RELAXED: |     | |     |   |   `-GetBoundExpr {{.+}} upper
// RELAXED: |     | |     |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED: |     | |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <NoOp>
// RELAXED: |     | |     |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     | |     `-BinaryOperator {{.+}} 'int' '<='
// RELAXED: |     | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// RELAXED: |     | |       | `-IntegerLiteral {{.+}} 0
// RELAXED: |     | |       `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
// RELAXED: |     | |-OpaqueValueExpr [[ove]]
// RELAXED: |     | | `-ImplicitCastExpr {{.+}} 'char *' <LValueToRValue>
// RELAXED: |     | |   `-DeclRefExpr {{.+}} [[var_ptr]]
// RELAXED: |     | |-OpaqueValueExpr [[ove_1]]
// RELAXED: |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// RELAXED: |     | |   `-DeclRefExpr {{.+}} [[var_bidi]]
// RELAXED: |     | `-OpaqueValueExpr [[ove_2]]
// RELAXED: |     |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// RELAXED: |     |     `-IntegerLiteral {{.+}} 5
// RELAXED: |     |-OpaqueValueExpr [[ove]] {{.*}} 'char *'
// RELAXED: |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__bidi_indexable'
// RELAXED: |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'long'
//
// STRICT: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// STRICT: | |-ParmVarDecl [[var_ptr:0x[^ ]+]]
// STRICT: | |-ParmVarDecl [[var_bidi:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |     `-RecoveryExpr
// STRICT: |       |-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// STRICT: |       |-DeclRefExpr {{.+}} [[var_ptr]]
// STRICT: |       `-DeclRefExpr {{.+}} [[var_bidi]]

// RELAXED: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// RELAXED:   |-ParmVarDecl [[var_ptr_1:0x[^ ]+]]
// RELAXED:   |-ParmVarDecl [[var_bidi_1:0x[^ ]+]]
// RELAXED:   `-CompoundStmt
// RELAXED:     `-CallExpr
// RELAXED:       |-ImplicitCastExpr {{.+}} 'void (*__single)(char *, char *__bidi_indexable)' <FunctionToPointerDecay>
// RELAXED:       | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// RELAXED:       |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// RELAXED:       | `-ImplicitCastExpr {{.+}} 'char *__unsafe_indexable' <LValueToRValue>
// RELAXED:       |   `-DeclRefExpr {{.+}} [[var_ptr_1]]
// RELAXED:       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// RELAXED:         `-DeclRefExpr {{.+}} [[var_bidi_1]]
//
// STRICT: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// STRICT:   |-ParmVarDecl [[var_ptr_1:0x[^ ]+]]
// STRICT:   |-ParmVarDecl [[var_bidi_1:0x[^ ]+]]
// STRICT:   `-CompoundStmt
// STRICT:     `-CallExpr
// STRICT:       |-ImplicitCastExpr {{.+}} 'void (*__single)(char *, char *__bidi_indexable)' <FunctionToPointerDecay>
// STRICT:       | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// STRICT:       |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// STRICT:       | `-ImplicitCastExpr {{.+}} 'char *__unsafe_indexable' <LValueToRValue>
// STRICT:       |   `-DeclRefExpr {{.+}} [[var_ptr_1]]
// STRICT:       `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// STRICT:         `-DeclRefExpr {{.+}} [[var_bidi_1]]

