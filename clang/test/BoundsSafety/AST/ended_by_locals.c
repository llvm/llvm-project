

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s | FileCheck %s
#include <ptrcheck.h>

void foo(int * __bidi_indexable asdf, int asdf_len) {
    const int *myEndPtr = asdf + asdf_len;
    const int * __ended_by(myEndPtr) myEndedByPtr = asdf;
}

// CHECK:FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK:|-ParmVarDecl [[var_asdf:0x[^ ]+]]
// CHECK:|-ParmVarDecl [[var_asdf_len:0x[^ ]+]]
// CHECK:`-CompoundStmt
// CHECK:  |-DeclStmt
// CHECK:  | `-VarDecl [[var_myEndPtr:0x[^ ]+]]
// CHECK:  |   `-ImplicitCastExpr {{.+}} 'const int *__single /* __started_by(myEndedByPtr) */ ':'const int *__single' <BoundsSafetyPointerCast>
// CHECK:  |     `-BoundsCheckExpr {{.+}} 'asdf + asdf_len <= __builtin_get_pointer_upper_bound(asdf + asdf_len) && __builtin_get_pointer_lower_bound(asdf + asdf_len) <= asdf + asdf_len'
// CHECK:  |       |-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <NoOp>
// CHECK:  |       | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:  |       |   |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:  |       |   | `-DeclRefExpr {{.+}} [[var_asdf]]
// CHECK:  |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:  |       |     `-DeclRefExpr {{.+}} [[var_asdf_len]]
// CHECK:  |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:  |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:  |       | | |-ImplicitCastExpr {{.+}} 'const int *' <BoundsSafetyPointerCast>
// CHECK:  |       | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'const int *__bidi_indexable'
// CHECK:  |       | | `-GetBoundExpr {{.+}} upper
// CHECK:  |       | |   `-OpaqueValueExpr [[ove]] {{.*}} 'const int *__bidi_indexable'
// CHECK:  |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:  |       |   |-GetBoundExpr {{.+}} lower
// CHECK:  |       |   | `-OpaqueValueExpr [[ove]] {{.*}} 'const int *__bidi_indexable'
// CHECK:  |       |   `-ImplicitCastExpr {{.+}} 'const int *' <BoundsSafetyPointerCast>
// CHECK:  |       |     `-OpaqueValueExpr [[ove]] {{.*}} 'const int *__bidi_indexable'
// CHECK:  |       `-OpaqueValueExpr [[ove]]
// CHECK:  |         `-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <NoOp>
// CHECK:  |           `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK:  |             |-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:  |             | `-DeclRefExpr {{.+}} [[var_asdf]]
// CHECK:  |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:  |               `-DeclRefExpr {{.+}} [[var_asdf_len]]
// CHECK:  `-DeclStmt
// CHECK:    `-VarDecl [[var_myEndedByPtr:0x[^ ]+]]
// CHECK:      `-BoundsCheckExpr {{.+}} 'myEndPtr <= __builtin_get_pointer_upper_bound(asdf) && asdf <= myEndPtr && __builtin_get_pointer_lower_bound(asdf) <= asdf'
// CHECK:        |-ImplicitCastExpr {{.+}} 'const int *__single __ended_by(myEndPtr)':'const int *__single' <BoundsSafetyPointerCast>
// CHECK:        | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'const int *__bidi_indexable'
// CHECK:        |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:        | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:        | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:        | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'const int *'
// CHECK:        | | | `-GetBoundExpr {{.+}} upper
// CHECK:        | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'const int *__bidi_indexable'
// CHECK:        | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:        | |   |-ImplicitCastExpr {{.+}} 'const int *' <BoundsSafetyPointerCast>
// CHECK:        | |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'const int *__bidi_indexable'
// CHECK:        | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'const int *'
// CHECK:        | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:        |   |-GetBoundExpr {{.+}} lower
// CHECK:        |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'const int *__bidi_indexable'
// CHECK:        |   `-ImplicitCastExpr {{.+}} 'const int *' <BoundsSafetyPointerCast>
// CHECK:        |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'const int *__bidi_indexable'
// CHECK:        |-OpaqueValueExpr [[ove_1]]
// CHECK:        | `-ImplicitCastExpr {{.+}} 'const int *__bidi_indexable' <NoOp>
// CHECK:        |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:        |     `-DeclRefExpr {{.+}} [[var_asdf]]
// CHECK:        `-OpaqueValueExpr [[ove_2]]
// CHECK:          `-ImplicitCastExpr {{.+}} 'const int *' <BoundsSafetyPointerCast>
// CHECK:            `-BoundsSafetyPointerPromotionExpr {{.+}} 'const int *__bidi_indexable'
// CHECK:              |-DeclRefExpr {{.+}} [[var_myEndPtr]]
// CHECK:              |-ImplicitCastExpr {{.+}} 'const int *__single /* __started_by(myEndedByPtr) */ ':'const int *__single' <LValueToRValue>
// CHECK:              | `-DeclRefExpr {{.+}} [[var_myEndPtr]]
// CHECK:              `-ImplicitCastExpr {{.+}} 'const int *__single __ended_by(myEndPtr)':'const int *__single' <LValueToRValue>
// CHECK:                `-DeclRefExpr {{.+}} [[var_myEndedByPtr]]

