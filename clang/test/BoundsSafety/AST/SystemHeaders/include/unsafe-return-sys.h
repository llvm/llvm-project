#include <unsafe-global-ext.h>

// RELAXED: VarDecl [[var_sizedGlobal:0x[^ ]+]]
// RELAXED: VarDecl [[var_valueTerminatedGlobal:0x[^ ]+]]
// RELAXED: VarDecl [[var_bidiGlobal:0x[^ ]+]]

// STRICT: VarDecl [[var_sizedGlobal:0x[^ ]+]]
// STRICT: VarDecl [[var_valueTerminatedGlobal:0x[^ ]+]]
// STRICT: VarDecl [[var_bidiGlobal:0x[^ ]+]]

#pragma clang system_header

int * __unsafe_indexable funcInSDK(int * __unsafe_indexable unsafePointer) {
    return unsafePointer;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// RELAXED: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_unsafePointer]]

// STRICT: |-FunctionDecl [[func_funcInSDK:0x[^ ]+]] {{.+}} funcInSDK
// STRICT: | |-ParmVarDecl [[var_unsafePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafePointer]]

int * __unsafe_indexable funcInSDK2(int * __single __terminated_by(2) safePointer) {
    return safePointer;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// RELAXED: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_safePointer]]

// STRICT: |-FunctionDecl [[func_funcInSDK2:0x[^ ]+]] {{.+}} funcInSDK2
// STRICT: | |-ParmVarDecl [[var_safePointer:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_safePointer]]

int * __single funcInSDK3(int * __unsafe_indexable unsafePointer) {
    return unsafePointer; // strict-error{{returning 'int *__unsafe_indexable' from a function with incompatible result type 'int *__single' casts away '__unsafe_indexable' qualifier; use '__unsafe_forge_single' or '__unsafe_forge_bidi_indexable' to perform this conversion}}
}

// RELAXED: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// RELAXED: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]

// STRICT: |-FunctionDecl [[func_funcInSDK3:0x[^ ]+]] {{.+}} funcInSDK3
// STRICT: | |-ParmVarDecl [[var_unsafePointer_1:0x[^ ]+]]
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-RecoveryExpr
// STRICT: |       `-DeclRefExpr {{.+}} [[var_unsafePointer_1]]

int * __unsafe_indexable funcInSDK4(void) {
    return sizedGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// RELAXED: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// RELAXED: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// RELAXED: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// RELAXED: |         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |         | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// RELAXED: |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// RELAXED: |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// RELAXED: |         | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// RELAXED: |         | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |         | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// RELAXED: |         | |-OpaqueValueExpr [[ove]]
// RELAXED: |         | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |         | |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// RELAXED: |         | `-OpaqueValueExpr [[ove_1]]
// RELAXED: |         |   `-IntegerLiteral {{.+}} 2
// RELAXED: |         |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// RELAXED: |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'

// STRICT: |-FunctionDecl [[func_funcInSDK4:0x[^ ]+]] {{.+}} funcInSDK4
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// STRICT: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// STRICT: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// STRICT: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// STRICT: |         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |         | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// STRICT: |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// STRICT: |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// STRICT: |         | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// STRICT: |         | | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |         | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// STRICT: |         | |-OpaqueValueExpr [[ove]]
// STRICT: |         | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(2)':'int *__single' <LValueToRValue>
// STRICT: |         | |   `-DeclRefExpr {{.+}} [[var_sizedGlobal]]
// STRICT: |         | `-OpaqueValueExpr [[ove_1]]
// STRICT: |         |   `-IntegerLiteral {{.+}} 2
// STRICT: |         |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by(2)':'int *__single'
// STRICT: |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'

int * __unsafe_indexable funcInSDK5(void) {
    return valueTerminatedGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK5:0x[^ ]+]] {{.+}} funcInSDK5
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// RELAXED: |       `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK5:0x[^ ]+]] {{.+}} funcInSDK5
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// STRICT: |       `-DeclRefExpr {{.+}} [[var_valueTerminatedGlobal]]

int * __unsafe_indexable funcInSDK6(void) {
    return bidiGlobal;
}

// RELAXED: |-FunctionDecl [[func_funcInSDK6:0x[^ ]+]] {{.+}} funcInSDK6
// RELAXED: | `-CompoundStmt
// RELAXED: |   `-ReturnStmt
// RELAXED: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// RELAXED: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// RELAXED: |         `-DeclRefExpr {{.+}} [[var_bidiGlobal]]

// STRICT: |-FunctionDecl [[func_funcInSDK6:0x[^ ]+]] {{.+}} funcInSDK6
// STRICT: | `-CompoundStmt
// STRICT: |   `-ReturnStmt
// STRICT: |     `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <BoundsSafetyPointerCast>
// STRICT: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// STRICT: |         `-DeclRefExpr {{.+}} [[var_bidiGlobal]]

// MAINCHECK: `-FunctionDecl [[func_func:0x[^ ]+]] {{.+}} func
// MAINCHECK:   |-ParmVarDecl [[var_unsafe:0x[^ ]+]]
// MAINCHECK:   |-ParmVarDecl [[var_term:0x[^ ]+]]
// MAINCHECK:   `-CompoundStmt
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(int *__unsafe_indexable)' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_unsafe]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(int *__single __terminated_by(2))' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK2]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__single __terminated_by(2)':'int *__single' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_term]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | |-ImplicitCastExpr {{.+}} 'int *__single(*__single)(int *__unsafe_indexable)' <FunctionToPointerDecay>
// MAINCHECK:     | | `-DeclRefExpr {{.+}} [[func_funcInSDK3]]
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable' <LValueToRValue>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[var_unsafe]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(void)' <FunctionToPointerDecay>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[func_funcInSDK4]]
// MAINCHECK:     |-CallExpr
// MAINCHECK:     | `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(void)' <FunctionToPointerDecay>
// MAINCHECK:     |   `-DeclRefExpr {{.+}} [[func_funcInSDK5]]
// MAINCHECK:     `-CallExpr
// MAINCHECK:       `-ImplicitCastExpr {{.+}} 'int *__unsafe_indexable(*__single)(void)' <FunctionToPointerDecay>
// MAINCHECK:         `-DeclRefExpr {{.+}} [[func_funcInSDK6]]
