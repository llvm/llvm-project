#include <unsafe-inter-sysheader-other-sys.h>

// RELAXED: |-FunctionDecl [[func_funcWithAnnotation:0x[^ ]+]] {{.+}} funcWithAnnotation
// RELAXED: | `-ParmVarDecl [[var_foo:0x[^ ]+]]
// RELAXED: |-FunctionDecl [[func_funcWithoutAnnotation:0x[^ ]+]] {{.+}} funcWithoutAnnotation
// RELAXED: | `-ParmVarDecl [[var_foo_1:0x[^ ]+]]
// RELAXED: |-VarDecl [[var_safeGlobal:0x[^ ]+]]
// RELAXED: |-VarDecl [[var_unsafeGlobal:0x[^ ]+]]

// STRICT: |-FunctionDecl [[func_funcWithAnnotation:0x[^ ]+]] {{.+}} funcWithAnnotation
// STRICT: | `-ParmVarDecl [[var_foo:0x[^ ]+]]
// STRICT: |-FunctionDecl [[func_funcWithoutAnnotation:0x[^ ]+]] {{.+}} funcWithoutAnnotation
// STRICT: | `-ParmVarDecl [[var_foo_1:0x[^ ]+]]
// STRICT: VarDecl [[var_safeGlobal:0x[^ ]+]]
// STRICT: VarDecl [[var_unsafeGlobal:0x[^ ]+]]

#pragma clang system_header

void funcInSDK(int * unsafePointer) {
  // strict-error@+1{{assigning to 'int *__single' from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  safeGlobal = unsafePointer;
  unsafeGlobal = unsafePointer;
  unsafePointer = safeGlobal;
  unsafePointer = unsafeGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// RELAXED: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_safeGlobal]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |   |     `-DeclRefExpr {{.+}} [[var_unsafePointer]]
// RELAXED: |   |-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_unsafeGlobal]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer]]
// RELAXED: |   |-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   |     `-DeclRefExpr {{.+}} [[var_safeGlobal]]
// RELAXED: |   `-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_unsafePointer]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_unsafeGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// STRICT: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |   | `-RecoveryExpr
// STRICT: |   |   |-DeclRefExpr {{.+}} [[var_safeGlobal]]
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer]]
// STRICT: |   |-BinaryOperator {{.+}} 'int *' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_unsafeGlobal]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer]]
// STRICT: |   |-BinaryOperator {{.+}} 'int *' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   |     `-DeclRefExpr {{.+}} [[var_safeGlobal]]
// STRICT: |   `-BinaryOperator {{.+}} 'int *' '='
// STRICT: |     |-DeclRefExpr {{.+}} [[var_unsafePointer]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafeGlobal]]

void funcInSDK2(int * unsafePointer) {
  // strict-error@+1{{passing 'int *' to parameter of incompatible type 'int *__single __sized_by(4)' (aka 'int *__single') casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  funcWithAnnotation(unsafePointer);
  funcWithoutAnnotation(unsafePointer);
}

// RELAXED: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// RELAXED: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |   | | |-CallExpr
// RELAXED: |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(4))' <FunctionToPointerDecay>
// RELAXED: |   | | | | `-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// RELAXED: |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(4)':'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |   | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *'
// RELAXED: |   | | `-OpaqueValueExpr [[ove]]
// RELAXED: |   | |   `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |   | |     `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// RELAXED: |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *'
// RELAXED: |   `-CallExpr
// RELAXED: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// RELAXED: |     | `-DeclRefExpr {{.+}} [[func_funcWithoutAnnotation]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]

// STRICT: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// STRICT: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |   | `-RecoveryExpr
// STRICT: |   |   |-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// STRICT: |   `-CallExpr
// STRICT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// STRICT: |     | `-DeclRefExpr {{.+}} [[func_funcWithoutAnnotation]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]

// strict-note@+1{{passing argument to parameter 'safePointer' here}}
void funcInSDK3(int * __single safePointer) {
  safeGlobal = safePointer;
  unsafeGlobal = safePointer;
  safePointer = safeGlobal;
  // strict-error@+1{{assigning to 'int *__single' from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  safePointer = unsafeGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// RELAXED: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_safeGlobal]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   |   `-DeclRefExpr {{.+}} [[var_safePointer]]
// RELAXED: |   |-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_unsafeGlobal]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   |     `-DeclRefExpr {{.+}} [[var_safePointer]]
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_safePointer]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   |   `-DeclRefExpr {{.+}} [[var_safeGlobal]]
// RELAXED: |   `-BinaryOperator {{.+}} 'int *__single' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_safePointer]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_unsafeGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// STRICT: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-BinaryOperator {{.+}} 'int *__single' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_safeGlobal]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_safePointer]]
// STRICT: |   |-BinaryOperator {{.+}} 'int *' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_unsafeGlobal]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   |     `-DeclRefExpr {{.+}} [[var_safePointer]]
// STRICT: |   |-BinaryOperator {{.+}} 'int *__single' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_safePointer]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_safeGlobal]]
// STRICT: |   `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |     `-RecoveryExpr
// STRICT: |       |-DeclRefExpr {{.+}} [[var_safePointer]]
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafeGlobal]]

void funcInSDK4(int * __single safePointer) {
  funcWithAnnotation(safePointer);
  funcWithoutAnnotation(safePointer);
}

// RELAXED: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// RELAXED: | |-ParmVarDecl [[var_safePointer_1:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |   | | |-CallExpr
// RELAXED: |   | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(4))' <FunctionToPointerDecay>
// RELAXED: |   | | | | `-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// RELAXED: |   | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__single'
// RELAXED: |   | | `-OpaqueValueExpr [[ove_1]]
// RELAXED: |   | |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   | |     `-DeclRefExpr {{.+}} [[var_safePointer_1]]
// RELAXED: |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single'
// RELAXED: |   `-CallExpr
// RELAXED: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// RELAXED: |     | `-DeclRefExpr {{.+}} [[func_funcWithoutAnnotation]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_safePointer_1]]

// STRICT: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// STRICT: | |-ParmVarDecl [[var_safePointer_1:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// STRICT: |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// STRICT: |   | | |-BoundsCheckExpr {{.+}} 'safePointer <= __builtin_get_pointer_upper_bound(safePointer) && __builtin_get_pointer_lower_bound(safePointer) <= safePointer && 4 <= (char *)__builtin_get_pointer_upper_bound(safePointer) - (char *)safePointer && 0 <= 4'
// STRICT: |   | | | |-CallExpr
// STRICT: |   | | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(4))' <FunctionToPointerDecay>
// STRICT: |   | | | | | `-DeclRefExpr {{.+}} [[func_funcWithAnnotation]]
// STRICT: |   | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single'
// STRICT: |   | | | `-BinaryOperator {{.+}} 'int' '&&'
// STRICT: |   | | |   |-BinaryOperator {{.+}} 'int' '&&'
// STRICT: |   | | |   | |-BinaryOperator {{.+}} 'int' '<='
// STRICT: |   | | |   | | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   | | |   | |   `-GetBoundExpr {{.+}} upper
// STRICT: |   | | |   | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// STRICT: |   | | |   | |       `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | | |   | `-BinaryOperator {{.+}} 'int' '<='
// STRICT: |   | | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   | | |   |   | `-GetBoundExpr {{.+}} lower
// STRICT: |   | | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// STRICT: |   | | |   |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | | |   `-BinaryOperator {{.+}} 'int' '&&'
// STRICT: |   | | |     |-BinaryOperator {{.+}} 'int' '<='
// STRICT: |   | | |     | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'long'
// STRICT: |   | | |     | `-BinaryOperator {{.+}} 'long' '-'
// STRICT: |   | | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// STRICT: |   | | |     |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// STRICT: |   | | |     |   |   `-GetBoundExpr {{.+}} upper
// STRICT: |   | | |     |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// STRICT: |   | | |     |   |       `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | | |     |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   | | |     |     `-DeclRefExpr {{.+}} 'int *__single' lvalue ParmVar {{.+}} 'safePointer' 'int *__single'
// STRICT: |   | | |     `-BinaryOperator {{.+}} 'int' '<='
// STRICT: |   | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// STRICT: |   | | |       | `-IntegerLiteral {{.+}} 0
// STRICT: |   | | |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'long'
// STRICT: |   | | |-OpaqueValueExpr [[ove]]
// STRICT: |   | | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   | | |   `-DeclRefExpr {{.+}} [[var_safePointer_1]]
// STRICT: |   | | `-OpaqueValueExpr [[ove_1]]
// STRICT: |   | |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// STRICT: |   | |     `-IntegerLiteral {{.+}} 4
// STRICT: |   | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single'
// STRICT: |   | `-OpaqueValueExpr [[ove_1]] {{.*}} 'long'
// STRICT: |   `-CallExpr
// STRICT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// STRICT: |     | `-DeclRefExpr {{.+}} [[func_funcWithoutAnnotation]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |       `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |         `-DeclRefExpr {{.+}} [[var_safePointer_1]]

void funcInSDK5(int * unsafePointer) {
  funcInSDK(unsafePointer);
  // strict-error@+1{{passing 'int *' to parameter of incompatible type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  funcInSDK3(unsafePointer);
}

// RELAXED: |-FunctionDecl [[func_funcInSDK5:0x[^ ]+]] {{.+}} funcInSDK5
// RELAXED: | |-ParmVarDecl [[var_unsafePointer_2:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-CallExpr
// RELAXED: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// RELAXED: |   | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer_2]]
// RELAXED: |   `-CallExpr
// RELAXED: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single)' <FunctionToPointerDecay>
// RELAXED: |     | `-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_unsafePointer_2]]

// STRICT: |-FunctionDecl [[func_funcInSDK5:0x[^ ]+]] {{.+}} funcInSDK5
// STRICT: | |-ParmVarDecl [[var_unsafePointer_2:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-CallExpr
// STRICT: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// STRICT: |   | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer_2]]
// STRICT: |   `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |     `-RecoveryExpr
// STRICT: |       |-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafePointer_2]]

void funcInSDK6(int * __single safePointer) {
  funcInSDK(safePointer);
  funcInSDK3(safePointer);
}

// RELAXED: |-FunctionDecl [[func_funcInSDK6:0x[^ ]+]] {{.+}} funcInSDK6
// RELAXED: | |-ParmVarDecl [[var_safePointer_2:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-CallExpr
// RELAXED: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// RELAXED: |   | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |   |     `-DeclRefExpr {{.+}} [[var_safePointer_2]]
// RELAXED: |   `-CallExpr
// RELAXED: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single)' <FunctionToPointerDecay>
// RELAXED: |     | `-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_safePointer_2]]

// STRICT: |-FunctionDecl [[func_funcInSDK6:0x[^ ]+]] {{.+}} funcInSDK6
// STRICT: | |-ParmVarDecl [[var_safePointer_2:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-CallExpr
// STRICT: |   | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// STRICT: |   | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |   |     `-DeclRefExpr {{.+}} [[var_safePointer_2]]
// STRICT: |   `-CallExpr
// STRICT: |     |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single)' <FunctionToPointerDecay>
// STRICT: |     | `-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_safePointer_2]]

