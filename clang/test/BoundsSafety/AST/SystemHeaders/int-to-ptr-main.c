
#include <int-to-ptr-sys.h>

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=both -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix RELAXED
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=both -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix RELAXED
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix STRICT
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s -verify=strict,both -fno-bounds-safety-relaxed-system-headers -I %S/include -x objective-c -fexperimental-bounds-safety-objc | FileCheck %s --implicit-check-not "GetBoundExpr {{.+}} 'char *__single'" --implicit-check-not "GetBoundExpr {{.+}} 'char *'" --check-prefix STRICT

int * func(intptr_t y) {
  // both-error@+1{{returning 'int *' from a function with incompatible result type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  return funcSDK(y);
}

// RELAXED: |-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// RELAXED: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_p]]
//
// STRICT: |-FunctionDecl [[func_static:0x[^ ]+]] {{.+}} static
// STRICT: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_p]]


// RELAXED: |-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// RELAXED: | |-ParmVarDecl [[var_x:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-IfStmt
// RELAXED: |     |-BinaryOperator {{.+}} 'intptr_t':'long' '%'
// RELAXED: |     | |-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// RELAXED: |     | | `-DeclRefExpr {{.+}} [[var_x]]
// RELAXED: |     | `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <IntegralCast>
// RELAXED: |     |   `-IntegerLiteral {{.+}} 128
// RELAXED: |     |-ReturnStmt
// RELAXED: |     | `-RecoveryExpr
// RELAXED: |     |   |-DeclRefExpr {{.+}} [[func_static]]
// RELAXED: |     |   `-DeclRefExpr {{.+}} [[var_x]]
// RELAXED: |     `-ReturnStmt
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |         `-CallExpr
// RELAXED: |           |-ImplicitCastExpr {{.+}} 'int *__single(*__single)(int *__single)' <FunctionToPointerDecay>
// RELAXED: |           | `-DeclRefExpr {{.+}} [[func_static]]
// RELAXED: |           `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |             `-CStyleCastExpr {{.+}} 'int *' <IntegralToPointer>
// RELAXED: |               `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// RELAXED: |                 `-DeclRefExpr {{.+}} [[var_x]]
//
// STRICT: |-FunctionDecl [[func_static_1:0x[^ ]+]] {{.+}} static
// STRICT: | |-ParmVarDecl [[var_x:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-IfStmt
// STRICT: |     |-BinaryOperator {{.+}} 'intptr_t':'long' '%'
// STRICT: |     | |-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// STRICT: |     | | `-DeclRefExpr {{.+}} [[var_x]]
// STRICT: |     | `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <IntegralCast>
// STRICT: |     |   `-IntegerLiteral {{.+}} 128
// STRICT: |     |-ReturnStmt
// STRICT: |     | `-RecoveryExpr
// STRICT: |     |   |-DeclRefExpr {{.+}} [[func_static]]
// STRICT: |     |   `-DeclRefExpr {{.+}} [[var_x]]
// STRICT: |     `-ReturnStmt
// STRICT: |       `-RecoveryExpr
// STRICT: |         |-DeclRefExpr {{.+}} [[func_static]]
// STRICT: |         `-CStyleCastExpr {{.+}} 'int *' <IntegralToPointer>
// STRICT: |           `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// STRICT: |             `-DeclRefExpr {{.+}} [[var_x]]

// RELAXED: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// RELAXED:   |-ParmVarDecl [[var_y:0x[^ ]+]]
// RELAXED:   `-CompoundStmt
// RELAXED:     `-ReturnStmt
// RELAXED:       `-RecoveryExpr
// RELAXED:         `-CallExpr
// RELAXED:           |-ImplicitCastExpr {{.+}} 'int *(*__single)(intptr_t)' <FunctionToPointerDecay>
// RELAXED:           | `-DeclRefExpr {{.+}} [[func_static_1]]
// RELAXED:           `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// RELAXED:             `-DeclRefExpr {{.+}} [[var_y]]
//
// STRICT: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// STRICT:   |-ParmVarDecl [[var_y:0x[^ ]+]]
// STRICT:   `-CompoundStmt
// STRICT:     `-ReturnStmt
// STRICT:       `-RecoveryExpr
// STRICT:         `-CallExpr
// STRICT:           |-ImplicitCastExpr {{.+}} 'int *(*__single)(intptr_t)' <FunctionToPointerDecay>
// STRICT:           | `-DeclRefExpr {{.+}} [[func_static_1]]
// STRICT:           `-ImplicitCastExpr {{.+}} 'intptr_t':'long' <LValueToRValue>
// STRICT:             `-DeclRefExpr {{.+}} [[var_y]]
