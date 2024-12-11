#include <ptrcheck.h>

extern int *__sized_by(2) sizedGlobal;
extern int *__terminated_by(2) valueTerminatedGlobal;
extern int *__bidi_indexable bidiGlobal;

// RELAXED: |-VarDecl [[var_sizedGlobal:0x[^ ]+]]
// RELAXED: |-VarDecl [[var_valueTerminatedGlobal:0x[^ ]+]]
// RELAXED: |-VarDecl [[var_bidiGlobal:0x[^ ]+]]

// STRICT: VarDecl [[var_sizedGlobal:0x[^ ]+]]
// STRICT: VarDecl [[var_valueTerminatedGlobal:0x[^ ]+]]
// STRICT: VarDecl [[var_bidiGlobal:0x[^ ]+]]

#pragma clang system_header

void funcInSDK(int * unsafePointer) {
  sizedGlobal = unsafePointer; //strict-error{{assigning to 'int *__single __sized_by(2)' (aka 'int *__single') from incompatible type 'int *' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
  valueTerminatedGlobal = unsafePointer; //strict-error{{assigning to 'int *__single __terminated_by(2)' (aka 'int *__single') from incompatible type 'int *' is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  // This should result in an unsafe BoundsSafetyPointerCast rdar://99202425
}

// RELAXED: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// RELAXED: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single __sized_by(2)':'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *'
// RELAXED: |   `-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_unsafePointer]]
//
// STRICT: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// STRICT: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |   | `-RecoveryExpr
// STRICT: |   |   |-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_unsafePointer]]
// STRICT: |   `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |     `-RecoveryExpr
// STRICT: |       |-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafePointer]]

void funcInSDK2(int * __single __terminated_by(2) safePointer) {
  sizedGlobal = safePointer; //strict-error{{assigning to 'int *__single __sized_by(2)' (aka 'int *__single') from incompatible type 'int *__single __terminated_by(2)' (aka 'int *__single') requires a linear search for the terminator; use '__terminated_by_to_indexable()' to perform this conversion explicitly}}
  valueTerminatedGlobal = safePointer;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// RELAXED: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single __sized_by(2)':'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// RELAXED: |   | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__single __terminated_by(2)':'int *__single'
// RELAXED: |   `-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_safePointer]]

// STRICT: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// STRICT: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |   | `-RecoveryExpr
// STRICT: |   |   |-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_safePointer]]
// STRICT: |   `-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// STRICT: |     |-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_safePointer]]

void funcInSDK3(int * unsafePointer) {
  unsafePointer = sizedGlobal;
  unsafePointer = valueTerminatedGlobal;
  unsafePointer = bidiGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// RELAXED: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// RELAXED: |   |     | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// RELAXED: |   |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// RELAXED: |   |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// RELAXED: |   |     | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |     | | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |     | | |   `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// RELAXED: |   |     | |-OpaqueValueExpr [[ove_2]]
// RELAXED: |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |   |     | |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// RELAXED: |   |     | `-OpaqueValueExpr [[ove_3]]
// RELAXED: |   |     |   `-IntegerLiteral {{.+}} 2
// RELAXED: |   |     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// RELAXED: |   |-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// RELAXED: |   | `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |   |   `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// RELAXED: |   `-BinaryOperator {{.+}} 'int *' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_bidiGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// STRICT: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-BinaryOperator {{.+}} 'int *' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// STRICT: |   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// STRICT: |   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// STRICT: |   |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |   |     | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// STRICT: |   |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// STRICT: |   |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// STRICT: |   |     | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |     | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |   |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// STRICT: |   |     | |-OpaqueValueExpr [[ove]]
// STRICT: |   |     | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <LValueToRValue>
// STRICT: |   |     | |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// STRICT: |   |     | `-OpaqueValueExpr [[ove_1]]
// STRICT: |   |     |   `-IntegerLiteral {{.+}} 2
// STRICT: |   |     |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |   |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// STRICT: |   |-BinaryOperator {{.+}} 'int *' '='
// STRICT: |   | |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// STRICT: |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |   |   `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// STRICT: |   |     `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]
// STRICT: |   `-BinaryOperator {{.+}} 'int *' '='
// STRICT: |     |-DeclRefExpr {{.+}} [[var_unsafePointer_1]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// STRICT: |         `-DeclRefExpr {{.+}} [[var_bidiGlobal]]

void funcInSDK4(int * __single __terminated_by(2) safePointer) {
  safePointer = sizedGlobal; //strict-error{{assigning to 'int *__single __terminated_by(2)' (aka 'int *__single') from incompatible type 'int *__single __sized_by(2)' (aka 'int *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  safePointer = valueTerminatedGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// RELAXED: | |-ParmVarDecl [[var_safePointer_1:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   |-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// RELAXED: |   | |-DeclRefExpr {{.+}} [[var_safePointer_1]]
// RELAXED: |   | `-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |   |   |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |   |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// RELAXED: |   |   | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |   | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// RELAXED: |   |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// RELAXED: |   |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// RELAXED: |   |   | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |   |   | | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |   | | |   `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// RELAXED: |   |   | |-OpaqueValueExpr [[ove_4]]
// RELAXED: |   |   | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |   |   | |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// RELAXED: |   |   | `-OpaqueValueExpr [[ove_5]]
// RELAXED: |   |   |   `-IntegerLiteral {{.+}} 2
// RELAXED: |   |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// RELAXED: |   `-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// RELAXED: |     |-DeclRefExpr {{.+}} [[var_safePointer_1]]
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// STRICT: | |-ParmVarDecl [[var_safePointer_1:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// STRICT: |   | `-RecoveryExpr
// STRICT: |   |   |-DeclRefExpr {{.+}} [[var_safePointer_1]]
// STRICT: |   |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// STRICT: |   `-BinaryOperator {{.+}} 'int *__single __terminated_by(2)':'int *__single' '='
// STRICT: |     |-DeclRefExpr {{.+}} [[var_safePointer_1]]
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]

// MAINCHECK: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// MAINCHECK:   |-ParmVarDecl [[var_unsafe:0x[^ ]+]]
// MAINCHECK:   |-ParmVarDecl [[var_term:0x[^ ]+]]
// MAINCHECK:   `-CompoundStmt
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_unsafe]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __terminated_by(2))' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK2]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_term]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *)' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_unsafe]]
// MAINCHECK:     `-CallExpr
// MAINCHECK:       |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __terminated_by(2))' <FunctionToPointerDecay>
// MAINCHECK:       | `-DeclRefExpr {{.+}} [[func_funcInSDK4]]
// MAINCHECK:       `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// MAINCHECK:         `-DeclRefExpr {{.+}} [[var_term]]

