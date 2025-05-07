
// RUN: %clang_cc1 -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -ast-dump  %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

void Test(int sel) {
    int a;
    int *x = &a;
    int *y = &a;
    int *z = sel ? x : y;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} used z 'int *__bidi_indexable' cinit
    // CHECK: |   `-ConditionalOperator {{.*}} 'int *__bidi_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'y' 'int *__bidi_indexable'
    

    char c;
    char *x_char = &c;
    char *y_char = &c;
    char *z_char = sel ? x_char : y_char;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} used z_char 'char *__bidi_indexable' cinit
    // CHECK: |   `-ConditionalOperator {{.*}} 'char *__bidi_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'char *__bidi_indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'x_char' 'char *__bidi_indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'char *__bidi_indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'y_char' 'char *__bidi_indexable'


    void *__indexable x_ix_void = &a;
    void *__indexable y_ix_void = &a;
    void *__indexable z_ix_void = x_ix_void ?: y_ix_void;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} used z_ix_void 'void *__indexable' cinit
    // CHECK: |   `-BinaryConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |     |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |     |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    int *__indexable x_ix = &a;
    int *__indexable y_ix = &a;
    int *__indexable z_ix = sel ? x_ix : y_ix;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} used z_ix 'int *__indexable' cinit
    // CHECK: |   `-ConditionalOperator{{.*}} 'int *__indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'x_ix' 'int *__indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'y_ix' 'int *__indexable'

    int *__single x_sg = &a;
    int *__single y_sg = &a;
    int *__single z_sg = sel ? x_sg : y_sg;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} z_sg 'int *__single' cinit
    // CHECK: |   `-ConditionalOperator {{.*}} 'int *__single'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'y_sg' 'int *__single'

    int *__unsafe_indexable x_uix = &a;
    int *__unsafe_indexable y_uix = &a;
    int *__unsafe_indexable z_uix = sel ? x_uix : y_uix;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} z_uix 'int *__unsafe_indexable' cinit
    // CHECK: |   `-ConditionalOperator {{.*}} 'int *__unsafe_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'x_uix' 'int *__unsafe_indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__unsafe_indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__unsafe_indexable' lvalue Var {{.*}} 'y_uix' 'int *__unsafe_indexable'

    char *__unsafe_indexable x_uix_char = &c;
    char *__unsafe_indexable y_uix_char = &c;
    char *__unsafe_indexable z_uix_char = sel ? x_uix_char : y_uix_char;
    // CHECK: |-DeclStmt
    // CHECK: | `-VarDecl {{.*}} z_uix_char 'char *__unsafe_indexable' cinit
    // CHECK: |   `-ConditionalOperator {{.*}} 'char *__unsafe_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'char *__unsafe_indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'char *__unsafe_indexable' lvalue Var {{.*}} 'x_uix_char' 'char *__unsafe_indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'char *__unsafe_indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'char *__unsafe_indexable' lvalue Var {{.*}} 'y_uix_char' 'char *__unsafe_indexable'

    z = sel ? x : y;
    // CHECK: |-BinaryOperator {{.*}} 'int *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'z' 'int *__bidi_indexable'
    // CHECK: | `-ConditionalOperator {{.*}} 'int *__bidi_indexable'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |   `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |     `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'y' 'int *__bidi_indexable'
    
    z = sel ? x : y_ix;
    // CHECK: |-BinaryOperator {{.*}} 'int *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'z' 'int *__bidi_indexable'
    // CHECK: | `-ConditionalOperator {{.*}} 'int *__bidi_indexable'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |   `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'y_ix' 'int *__indexable'
    
    z = x ?: y_sg;
    // CHECK: |-BinaryOperator {{.*}} 'int *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'z' 'int *__bidi_indexable'
    // CHECK: | `-BinaryConditionalOperator {{.*}} 'int *__bidi_indexable'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |   |-OpaqueValueExpr {{.*}} 'int *__bidi_indexable'
    // CHECK: |   | `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |   |   `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |   |-OpaqueValueExpr {{.*}} 'int *__bidi_indexable'
    // CHECK: |   | `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |   |   `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |   `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'y_sg' 'int *__single'
    
    z = x ?: y_ix_void;
    // CHECK: |-BinaryOperator {{.*}} 'int *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'z' 'int *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <BitCast>
    // CHECK: |   `-BinaryConditionalOperator {{.*}} 'void *__bidi_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |     |-OpaqueValueExpr {{.*}} 'int *__bidi_indexable'
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BitCast>
    // CHECK: |     | `-OpaqueValueExpr {{.*}} 'int *__bidi_indexable'
    // CHECK: |     |   `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |     |     `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'x' 'int *__bidi_indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    z_ix = sel ? x_ix_void : y;
    // CHECK: |-BinaryOperator {{.*}} 'int *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'z_ix' 'int *__indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'void *__bidi_indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BitCast>
    // CHECK: |         `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |           `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'y' 'int *__bidi_indexable'

    z_ix = sel ? x_sg : y_ix;
    // CHECK: |-BinaryOperator {{.*}} 'int *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'z_ix' 'int *__indexable'
    // CHECK: | `-ConditionalOperator {{.*}} 'int *__indexable'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |   | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |   |-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |   | `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |   |   `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |   `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |     `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'y_ix' 'int *__indexable'

    z_ix = x_sg ?: y_ix_void;
    // CHECK: |-BinaryOperator {{.*}} 'int *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'z_ix' 'int *__indexable'
    // CHECK: | `-ImplicitCastExpr {{.*}} 'int *__indexable' <BitCast>
    // CHECK: |   `-BinaryConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |     |-OpaqueValueExpr {{.*}} 'int *__single'
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |     |   `-OpaqueValueExpr {{.*}} 'int *__single'
    // CHECK: |     |     `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |     |       `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    z_ix_void = sel ? x_sg : y;
    // CHECK: |-BinaryOperator {{.*}} 'void *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'z_ix_void' 'void *__indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'void *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'int *__bidi_indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'y' 'int *__bidi_indexable'

    z_ix_void = sel ? x_sg : y_ix;
    // CHECK: |-BinaryOperator {{.*}} 'void *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'z_ix_void' 'void *__indexable'
    // CHECK: | `-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |   `-ConditionalOperator {{.*}} 'int *__indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |       `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'y_ix' 'int *__indexable'

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wbounds-attributes-implicit-conversion-single-to-explicit-indexable"
    z_ix_void = sel ? x_sg : y_sg;
    #pragma clang diagnostic pop
    // CHECK: |-BinaryOperator {{.*}} 'void *__indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'z_ix_void' 'void *__indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'void *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'void *__single' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'int *__single'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'y_sg' 'int *__single'

    z_char = sel ? x_ix : y_ix_void;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'x_ix' 'int *__indexable'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    z_char = x_sg ?: y_ix_void;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK: |     `-BinaryConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       |-OpaqueValueExpr {{.*}} 'int *__single'
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |       |   `-OpaqueValueExpr {{.*}} 'int *__single'
    // CHECK: |       |     `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       |       `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    z_char = sel ? x_sg : y_ix_void;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |       |   `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |       |     `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'x_sg' 'int *__single'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'

    z_char = sel ? x_ix_void : y;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.*}} 'char *__bidi_indexable' <BitCast>
    // CHECK: |   `-ConditionalOperator {{.*}} 'void *__bidi_indexable'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |     | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |     |-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |     | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |     |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |     `-ImplicitCastExpr {{.*}} 'void *__bidi_indexable' <BitCast>
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'int *__bidi_indexable' <LValueToRValue>
    // CHECK: |         `-DeclRefExpr {{.*}} 'int *__bidi_indexable' lvalue Var {{.*}} 'y' 'int *__bidi_indexable'

    z_char = x_ix_void ?: y_ix;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK: |     `-BinaryConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |       |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |       |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK: |       | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |         `-ImplicitCastExpr {{.*}} 'int *__indexable' <LValueToRValue>
    // CHECK: |           `-DeclRefExpr {{.*}} 'int *__indexable' lvalue Var {{.*}} 'y_ix' 'int *__indexable'

    z_char = sel ? x_ix_void : y_sg;
    // CHECK: |-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK: | |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK: | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK: |   `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK: |     `-ConditionalOperator {{.*}} 'void *__indexable'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'int' lvalue ParmVar {{.*}} 'sel' 'int'
    // CHECK: |       |-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK: |       | `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK: |       `-ImplicitCastExpr {{.*}} 'void *__indexable' <BitCast>
    // CHECK: |         `-ImplicitCastExpr {{.*}} 'int *__indexable' <BoundsSafetyPointerCast>
    // CHECK: |           `-ImplicitCastExpr {{.*}} 'int *__single' <LValueToRValue>
    // CHECK: |             `-DeclRefExpr {{.*}} 'int *__single' lvalue Var {{.*}} 'y_sg' 'int *__single'

    z_char = x_ix_void ?: y_ix_void;
    // CHECK: `-BinaryOperator {{.*}} 'char *__bidi_indexable' '='
    // CHECK:   |-DeclRefExpr {{.*}} 'char *__bidi_indexable' lvalue Var {{.*}} 'z_char' 'char *__bidi_indexable'
    // CHECK:   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
    // CHECK:     `-ImplicitCastExpr {{.+}} 'char *__indexable' <BitCast>
    // CHECK:       `-BinaryConditionalOperator {{.*}} 'void *__indexable'
    // CHECK:         |-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK:         | `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK:         |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK:         | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK:         |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK:         |-OpaqueValueExpr {{.*}} 'void *__indexable'
    // CHECK:         | `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK:         |   `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'x_ix_void' 'void *__indexable'
    // CHECK:         `-ImplicitCastExpr {{.*}} 'void *__indexable' <LValueToRValue>
    // CHECK:           `-DeclRefExpr {{.*}} 'void *__indexable' lvalue Var {{.*}} 'y_ix_void' 'void *__indexable'
}
