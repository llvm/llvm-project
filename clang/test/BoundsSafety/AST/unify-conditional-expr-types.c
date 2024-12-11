

// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify %s | FileCheck %s
#include <ptrcheck.h>

typedef char * __terminated_by('\0') A;
typedef char * __terminated_by((char)0) B;

A funcA();
B funcB();

char testFunc(int pred) {
    A foo = pred ? funcA() : funcB();
    return foo[0];
}

// CHECK: |-FunctionDecl [[func_funcA:0x[^ ]+]] {{.+}} funcA
// CHECK: |-FunctionDecl [[func_funcB:0x[^ ]+]] {{.+}} funcB
// CHECK: |-FunctionDecl [[func_testFunc:0x[^ ]+]] {{.+}} testFunc
// CHECK: | |-ParmVarDecl [[var_pred:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_foo:0x[^ ]+]]
// CHECK: |   |   `-ConditionalOperator {{.+}}'char *__single __terminated_by('\x00')':'char *__single'
// bHECK: |   |     |-ImplicitCastExpr {{.+}} 'bool' <IntegralToBoolean>
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     |   `-DeclRefExpr {{.+}} [[var_pred]]
// CHECK: |   |     |-CallExpr
// CHECK: |   |     | `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by('\x00')(*__single)()' <FunctionToPointerDecay>
// CHECK: |   |     |  `-DeclRefExpr {{.+}} [[func_funcA]]
// CHECK: |   |     `-CallExpr
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by((char)0)(*__single)()' <FunctionToPointerDecay>
// CHECK: |   |         `-DeclRefExpr {{.+}} [[func_funcB]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} *__single' <LValueToRValue>
// CHECK: |         | `-DeclRefExpr {{.+}} [[var_foo]]
// CHECK: |         `-IntegerLiteral {{.+}} 0

char testFunc2(int pred, char * __counted_by(cc) c, char * __single d, int cc, int dc) {
    char * __counted_by(5) foo = pred ? c : d;
    return foo[3];
}
// CHECK: |-FunctionDecl [[func_testFunc2:0x[^ ]+]] {{.+}} testFunc2
// CHECK: | |-ParmVarDecl [[var_pred_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_cc:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_dc:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_foo_1:0x[^ ]+]]
// CHECK: |   |   `-BoundsCheckExpr
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(5)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |   |     | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     |     | | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK: |   |     |     | | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |     | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     | | | `-GetBoundExpr {{.+}} upper
// CHECK: |   |     | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     | |   |-GetBoundExpr {{.+}} lower
// CHECK: |   |     | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     | |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |     | |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |   |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |   | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'long'
// CHECK: |   |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |   |     |   |   |-GetBoundExpr {{.+}} upper
// CHECK: |   |     |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |     |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |   |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |   |     |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |     |     | `-IntegerLiteral {{.+}} 0
// CHECK: |   |     |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: |   |     |-OpaqueValueExpr [[ove]]
// CHECK: |   |     | `-ConditionalOperator {{.+}} <col:34, col:45> 'char *__bidi_indexable'
// CHECK: |   |     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     |   | `-DeclRefExpr {{.+}} [[var_pred_1]]
// CHECK: |   |     |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |     |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |     |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |   |     |   | | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK: |   |     |   | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |   |     |   | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |   |     |   | | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK: |   |     |   | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |   |     |   | | |-OpaqueValueExpr [[ove_1]]
// CHECK: |   |     |   | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(cc)':'char *__single' <LValueToRValue>
// CHECK: |   |     |   | | |   `-DeclRefExpr {{.+}} [[var_c]]
// CHECK: |   |     |   | | `-OpaqueValueExpr [[ove_2]]
// CHECK: |   |     |   | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     |   | |     `-DeclRefExpr {{.+}} [[var_cc]]
// CHECK: |   |     |   | |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK: |   |     |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |   |     |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |     |     `-ImplicitCastExpr {{.+}} 'char *__single' <LValueToRValue>
// CHECK: |   |     |       `-DeclRefExpr {{.+}} [[var_d]]
// CHECK: |   |     `-OpaqueValueExpr [[ove_3]]
// CHECK: |   |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |   |         `-IntegerLiteral {{.+}} 5
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |         | | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'char *__single __counted_by(5)':'char *__single'
// CHECK: |         | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__single __counted_by(5)':'char *__single'
// CHECK: |         | | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | |-OpaqueValueExpr [[ove_4]]
// CHECK: |         | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(5)':'char *__single' <LValueToRValue>
// CHECK: |         | | |   `-DeclRefExpr {{.+}} [[var_foo_1]]
// CHECK: |         | | `-OpaqueValueExpr [[ove_5]]
// CHECK: |         | |   `-IntegerLiteral {{.+}} 5
// CHECK: |         | |-OpaqueValueExpr [[ove_4]] {{.*}} 'char *__single __counted_by(5)':'char *__single'
// CHECK: |         | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK: |         `-IntegerLiteral {{.+}} 3

char testFunc3(int pred, char * __ended_by(*f) * e, char * __ended_by(g) * f, char * g) {
    // expected-error@+1{{conditional expression evaluates values with incompatible nested pointer types 'char *__single __ended_by(*f)*__single' (aka 'char *__single*__single') and 'char *__single __ended_by(g) /* __started_by(*e) */ *__single' (aka 'char *__single*__single')}}
    char ** foo = pred ? e : f;
    const char * const bar = *foo;
    return bar[2];
}

// CHECK: |-FunctionDecl [[func_testFunc3:0x[^ ]+]] {{.+}} testFunc3
// CHECK: | |-ParmVarDecl [[var_pred_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_e:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_f:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_g:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_foo_2:0x[^ ]+]]
// CHECK: |   |   `-RecoveryExpr
// CHECK: |   |     |-DeclRefExpr {{.+}} [[var_pred_2]]
// CHECK: |   |     |-DeclRefExpr {{.+}} [[var_e]]
// CHECK: |   |     `-DeclRefExpr {{.+}} [[var_f]]
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_bar:0x[^ ]+]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} contains-errors <BoundsSafetyPointerCast>
// CHECK: |   |     `-ImplicitCastExpr {{.+}} contains-errors <NoOp>
// CHECK: |   |       `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// CHECK: |   |         `-UnaryOperator {{.+}} cannot overflow
// CHECK: |   |           `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// CHECK: |   |             `-DeclRefExpr {{.+}} [[var_foo_2]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// CHECK: |       `-ArraySubscriptExpr
// CHECK: |         |-ImplicitCastExpr {{.+}} contains-errors <LValueToRValue>
// CHECK: |         | `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK: |         `-IntegerLiteral {{.+}} 2

char externFunc(char * __counted_by(8));

// CHECK: |-FunctionDecl [[func_externFunc:0x[^ ]+]] {{.+}} externFunc

char testFunc4(int pred, char * __counted_by(7) * c, char * __counted_by(10) * d) {
    // expected-error@+1{{conditional expression evaluates values with incompatible nested pointer types 'char *__single __counted_by(7)*__single' (aka 'char *__single*__single') and 'char *__single __counted_by(10)*__single' (aka 'char *__single*__single')}}
    return externFunc(*(pred ? c : d));
}

// CHECK: |-FunctionDecl [[func_testFunc4:0x[^ ]+]] {{.+}} testFunc4
// CHECK: | |-ParmVarDecl [[var_pred_3:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d_1:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-CallExpr
// CHECK: |       |-DeclRefExpr {{.+}} [[func_externFunc]]
// CHECK: |       `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         `-ParenExpr
// CHECK: |           `-RecoveryExpr
// CHECK: |             |-DeclRefExpr {{.+}} [[var_pred_3]]
// CHECK: |             |-DeclRefExpr {{.+}} [[var_c_1]]
// CHECK: |             `-DeclRefExpr {{.+}} [[var_d_1]]

char testFunc5(int pred, char * __bidi_indexable * __counted_by(7) c, char * __bidi_indexable * __counted_by(10) d) {
    return externFunc((pred ? c : d)[8]);
}

// CHECK: |-FunctionDecl [[func_testFunc5:0x[^ ]+]] {{.+}} testFunc5
// CHECK: | |-ParmVarDecl [[var_pred_4:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |-BoundsCheckExpr
// CHECK: |       | | |-CallExpr
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'char (*__single)(char *__single __counted_by(8))' <FunctionToPointerDecay>
// CHECK: |       | | | | `-DeclRefExpr {{.+}} [[func_externFunc]]
// CHECK: |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(8)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |       | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | | |         |   | | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'char *__bidi_indexable*__single __counted_by(7)':'char *__bidi_indexable*__single'
// CHECK: |       | | |         |   | | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | |         |     | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |       | | |         |     | | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |       | |   | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |       | |   |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |     | |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'long'
// CHECK: |       | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |       | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |       | |     |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       | |       | `-IntegerLiteral {{.+}} 0
// CHECK: |       | |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'long'
// CHECK: |       | |-OpaqueValueExpr [[ove_10]]
// CHECK: |       | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: |       | |   `-ArraySubscriptExpr
// CHECK: |       | |     |-ParenExpr
// CHECK: |       | |     | `-ConditionalOperator {{.+}} <col:24, col:35> 'char *__bidi_indexable*__bidi_indexable'
// CHECK: |       | |     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |     |   | `-DeclRefExpr {{.+}} [[var_pred_4]]
// CHECK: |       | |     |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |     |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |     |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable*__bidi_indexable'
// CHECK: |       | |     |   | | | |-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable*__single __counted_by(7)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |   | | | |-BinaryOperator {{.+}} 'char *__bidi_indexable*' '+'
// CHECK: |       | |     |   | | | | |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*' <BoundsSafetyPointerCast>
// CHECK: |       | |     |   | | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable*__single __counted_by(7)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |   | | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK: |       | |     |   | | |-OpaqueValueExpr [[ove_11]]
// CHECK: |       | |     |   | | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__single __counted_by(7)':'char *__bidi_indexable*__single' <LValueToRValue>
// CHECK: |       | |     |   | | |   `-DeclRefExpr {{.+}} [[var_c_2]]
// CHECK: |       | |     |   | | `-OpaqueValueExpr [[ove_12]]
// CHECK: |       | |     |   | |   `-IntegerLiteral {{.+}} 7
// CHECK: |       | |     |   | |-OpaqueValueExpr [[ove_11]] {{.*}} 'char *__bidi_indexable*__single __counted_by(7)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |   | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK: |       | |     |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |     |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |     |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable*__bidi_indexable'
// CHECK: |       | |     |     | | |-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |     | | |-BinaryOperator {{.+}} 'char *__bidi_indexable*' '+'
// CHECK: |       | |     |     | | | |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*' <BoundsSafetyPointerCast>
// CHECK: |       | |     |     | | | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |     | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: |       | |     |     | |-OpaqueValueExpr [[ove_13]]
// CHECK: |       | |     |     | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single' <LValueToRValue>
// CHECK: |       | |     |     | |   `-DeclRefExpr {{.+}} [[var_d_2]]
// CHECK: |       | |     |     | `-OpaqueValueExpr [[ove_14]]
// CHECK: |       | |     |     |   `-IntegerLiteral {{.+}} 10
// CHECK: |       | |     |     |-OpaqueValueExpr [[ove_13]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |       | |     |     `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: |       | |     `-IntegerLiteral {{.+}} 8
// CHECK: |       | `-OpaqueValueExpr [[ove_15]]
// CHECK: |       |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       |     `-IntegerLiteral {{.+}} 8
// CHECK: |       |-OpaqueValueExpr [[ove_10]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'long'

char testFunc6(int pred, char * __bidi_indexable * __indexable c, char * __bidi_indexable * __counted_by(10) d) {
    char * __bidi_indexable *__indexable tmp = pred ? c : d;
    return externFunc(tmp[11]);
}

// CHECK: |-FunctionDecl [[func_testFunc6:0x[^ ]+]] {{.+}} testFunc6
// CHECK: | |-ParmVarDecl [[var_pred_5:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c_3:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d_3:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_tmp:0x[^ ]+]]
// CHECK: |   |   `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__indexable' <BoundsSafetyPointerCast>
// CHECK: |   |     `-ConditionalOperator {{.+}} <col:48, col:59> 'char *__bidi_indexable*__bidi_indexable'
// CHECK: |   |       |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |       | `-DeclRefExpr {{.+}} [[var_pred_5]]
// CHECK: |   |       |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |   |       | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__indexable' <LValueToRValue>
// CHECK: |   |       |   `-DeclRefExpr {{.+}} [[var_c_3]]
// CHECK: |   |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |   |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |   |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable*__bidi_indexable'
// CHECK: |   |         | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |   |         | | |-BinaryOperator {{.+}} 'char *__bidi_indexable*' '+'
// CHECK: |   |         | | | |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*' <BoundsSafetyPointerCast>
// CHECK: |   |         | | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |   |         | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK: |   |         | |-OpaqueValueExpr [[ove_16]]
// CHECK: |   |         | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single' <LValueToRValue>
// CHECK: |   |         | |   `-DeclRefExpr {{.+}} [[var_d_3]]
// CHECK: |   |         | `-OpaqueValueExpr [[ove_17]]
// CHECK: |   |         |   `-IntegerLiteral {{.+}} 10
// CHECK: |   |         |-OpaqueValueExpr [[ove_16]] {{.*}} 'char *__bidi_indexable*__single __counted_by(10)':'char *__bidi_indexable*__single'
// CHECK: |   |         `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK: |   `-ReturnStmt
// CHECK: |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |-BoundsCheckExpr
// CHECK: |       | | |-CallExpr
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'char (*__single)(char *__single __counted_by(8))' <FunctionToPointerDecay>
// CHECK: |       | | | | `-DeclRefExpr {{.+}} [[func_externFunc]]
// CHECK: |       | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(8)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |       | | |   `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |       | |   | |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |       | |   |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |     | |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'long'
// CHECK: |       | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |       | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |       | |     |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       | |       | `-IntegerLiteral {{.+}} 0
// CHECK: |       | |       `-OpaqueValueExpr [[ove_19]] {{.*}} 'long'
// CHECK: |       | |-OpaqueValueExpr [[ove_18]]
// CHECK: |       | | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK: |       | |   `-ArraySubscriptExpr
// CHECK: |       | |     |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: |       | |     | `-ImplicitCastExpr {{.+}} 'char *__bidi_indexable*__indexable' <LValueToRValue>
// CHECK: |       | |     |   `-DeclRefExpr {{.+}} [[var_tmp]]
// CHECK: |       | |     `-IntegerLiteral {{.+}} 11
// CHECK: |       | `-OpaqueValueExpr [[ove_19]]
// CHECK: |       |   `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       |     `-IntegerLiteral {{.+}} 8
// CHECK: |       |-OpaqueValueExpr [[ove_18]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       `-OpaqueValueExpr [[ove_19]] {{.*}} 'long'

char testFunc7(int pred, char * __unsafe_indexable * __single c, char * __single * __indexable d) {
    // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'char *__unsafe_indexable*__indexable' and 'char *__single*__indexable'; use explicit casts to perform this conversion}}
    char * __unsafe_indexable *__indexable tmp = pred ? c : d;
    return tmp[9][2];
}

char testFunc8(int pred, char * __ended_by(g) e, char * g) {
    // expected-error@+1{{local variable end must be declared right next to its dependent decl}}
    char *end = g;
    // expected-error@+2{{local variable foo must be declared right next to its dependent decl}}
    // expected-note@+1 2 {{previous use is here}}
    char * __ended_by(end) foo = e;
    // expected-error@+3{{local variable foo2 must be declared right next to its dependent decl}}
    // expected-error@+2{{variable 'end' referred to by __ended_by variable cannot be used in other dynamic bounds attributes}}
    // expected-note@+1{{previous use is here}}
    char * __ended_by(end) foo2 = foo + 1;
    // expected-error@+2{{local variable bar must be declared right next to its dependent decl}}
    // expected-error@+1 2 {{variable 'end' referred to by __ended_by variable cannot be used in other dynamic bounds attributes}}
    char * __ended_by(end) bar = *(pred ? &foo : &foo2);
    return bar[0];
}

char testFunc9(int pred, char * __counted_by(cc) c, char * __ended_by(e)  d, int cc, char *e) {
    char *foo = pred ? c : d;
    return foo[3];
}

// CHECK:  -FunctionDecl [[func_testFunc9:0x[^ ]+]] {{.+}} testFunc9
// CHECK:   |-ParmVarDecl [[var_pred_8:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_c_5:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_d_5:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_cc_1:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
// CHECK:   |-ParmVarDecl [[var_e_2:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_foo_4:0x[^ ]+]]
// CHECK:     |   `-ConditionalOperator {{.+}} <col:17, col:28> 'char *__bidi_indexable'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_pred_8]]
// CHECK:     |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     |     | | | |-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK:     |     | | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     |     | | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |     | | | | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK:     |     | | | | `-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'int'
// CHECK:     |     | | |-OpaqueValueExpr [[ove_27]]
// CHECK:     |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(cc)':'char *__single' <LValueToRValue>
// CHECK:     |     | | |   `-DeclRefExpr {{.+}} [[var_c_5]]
// CHECK:     |     | | `-OpaqueValueExpr [[ove_28]]
// CHECK:     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     | |     `-DeclRefExpr {{.+}} [[var_cc_1]]
// CHECK:     |     | |-OpaqueValueExpr [[ove_27]] {{.*}} 'char *__single __counted_by(cc)':'char *__single'
// CHECK:     |     | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int'
// CHECK:     |     `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:     |       |-DeclRefExpr {{.+}} [[var_d_5]]
// CHECK:     |       |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(d) */ ':'char *__single' <LValueToRValue>
// CHECK:     |       | `-DeclRefExpr {{.+}} [[var_e_2]]
// CHECK:     |       `-ImplicitCastExpr {{.+}} 'char *__single __ended_by(e)':'char *__single' <LValueToRValue>
// CHECK:     |         `-DeclRefExpr {{.+}} [[var_d_5]]
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK:         `-ArraySubscriptExpr
// CHECK:           |-ImplicitCastExpr {{.+}} 'char *__bidi_indexable' <LValueToRValue>
// CHECK:           | `-DeclRefExpr {{.+}} [[var_foo_4]]
// CHECK:           `-IntegerLiteral {{.+}} 3

// expected-error@+2{{'__counted_by' attribute on nested pointer type is only allowed on indirect parameters}}
// expected-error@+1{{'__ended_by' attribute on nested pointer type is only allowed on indirect parameters}}
char testFunc10(int pred, char * __counted_by(5) * __counted_by(cc) c, char * __ended_by(e) * __ended_by(f) d, int cc, char *e, char *f) {
    char **foo = pred ? c : d;
    // expected-error@+1{{array subscript on single pointer 'foo[3]' must use a constant index of 0 to be in bounds}}
    return foo[3][2];
}

char testFunc11(int pred, char *__null_terminated *__null_terminated *__terminated_by(0x42) * __counted_by(cc) c, char *__terminated_by(0x40) *__null_terminated *__terminated_by(0x42) * __ended_by(e) d, int cc, char *e) {
    // expected-error@+1{{conditional expression evaluates values with incompatible nested pointer types 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable' (aka 'char *__single*__single*__single*__bidi_indexable') and 'char *__single __terminated_by(64)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable' (aka 'char *__single*__single*__single*__bidi_indexable')}}
    char *__null_terminated *__null_terminated *__terminated_by(0x42) * foo = pred ? c : d;
    return ****foo;
}

char testFunc12(int pred, char *__null_terminated *__null_terminated *__terminated_by(0x42) * __counted_by(cc) c, char * *__null_terminated *__terminated_by(0x42) * __ended_by(e) d, int cc, char *e) {
    // expected-error@+1{{conditional expression evaluates values with incompatible nested pointer types 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable' (aka 'char *__single*__single*__single*__bidi_indexable') and 'char *__single*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable' (aka 'char *__single*__single*__single*__bidi_indexable')}}
    char *__single *__null_terminated *__terminated_by(0x42) * foo = pred ? c : d;
    return ****foo;
}

char testFunc13(int pred, char *__null_terminated *__null_terminated *__terminated_by(0x42) * __counted_by(cc) c, char *__null_terminated *__null_terminated *__terminated_by(0x42) * __ended_by(e) d, int cc, char *e) {
    char *__null_terminated *__null_terminated *__terminated_by(0x42) * foo = pred ? c : d;
    return ****foo;
}

// CHECK:  -FunctionDecl [[func_testFunc13:0x[^ ]+]] {{.+}} testFunc13
// CHECK:   |-ParmVarDecl [[var_pred_12:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_c_9:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_d_9:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_cc_5:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
// CHECK:   |-ParmVarDecl [[var_e_6:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_foo_8:0x[^ ]+]]
// CHECK:     |   `-ConditionalOperator {{.+}} <col:79, col:90> 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_pred_12]]
// CHECK:     |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable'
// CHECK:     |     | | | |-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single __counted_by(cc)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single'
// CHECK:     |     | | | |-BinaryOperator {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*' '+'
// CHECK:     |     | | | | |-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*' <BoundsSafetyPointerCast>
// CHECK:     |     | | | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single __counted_by(cc)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single'
// CHECK:     |     | | | | `-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'int'
// CHECK:     |     | | |-OpaqueValueExpr [[ove_28]]
// CHECK:     |     | | | `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single __counted_by(cc)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single' <LValueToRValue>
// CHECK:     |     | | |   `-DeclRefExpr {{.+}} [[var_c_9]]
// CHECK:     |     | | `-OpaqueValueExpr [[ove_29]]
// CHECK:     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     | |     `-DeclRefExpr {{.+}} [[var_cc_5]]
// CHECK:     |     | |-OpaqueValueExpr [[ove_28]] {{.*}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single __counted_by(cc)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single'
// CHECK:     |     | `-OpaqueValueExpr [[ove_29]] {{.*}} 'int'
// CHECK:     |     `-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable'
// CHECK:     |       |-DeclRefExpr {{.+}} [[var_d_9]]
// CHECK:     |       |-ImplicitCastExpr {{.+}} 'char *__single /* __started_by(d) */ ':'char *__single' <LValueToRValue>
// CHECK:     |       | `-DeclRefExpr {{.+}} [[var_e_6]]
// CHECK:     |       `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single __ended_by(e)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__single' <LValueToRValue>
// CHECK:     |         `-DeclRefExpr {{.+}} [[var_d_9]]
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK:         `-UnaryOperator {{.+}} cannot overflow
// CHECK:           `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)':'char *__single' <LValueToRValue>
// CHECK:             `-UnaryOperator {{.+}} cannot overflow
// CHECK:               `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)':'char *__single __terminated_by(0)*__single' <LValueToRValue>
// CHECK:                 `-UnaryOperator {{.+}} cannot overflow
// CHECK:                   `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)':'char *__single __terminated_by(0)*__single __terminated_by(0)*__single' <LValueToRValue>
// CHECK:                     `-UnaryOperator {{.+}} cannot overflow
// CHECK:                       `-ImplicitCastExpr {{.+}} 'char *__single __terminated_by(0)*__single __terminated_by(0)*__single __terminated_by(66)*__bidi_indexable' <LValueToRValue>
// CHECK:                         `-DeclRefExpr {{.+}} [[var_foo_8]]

char testFunc14(int pred, const char *__single _Nullable __null_terminated c, const char * _Nullable __null_terminated d) {
    const char *__null_terminated foo = pred ? c : d;
    return *foo;
}

// CHECK: |-FunctionDecl [[func_testFunc14:0x[^ ]+]] {{.+}} testFunc14
// CHECK: | |-ParmVarDecl [[var_pred_13:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c_10:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d_10:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_foo_9:0x[^ ]+]]
// CHECK: |   |   `-ConditionalOperator {{.+}} <col:41, col:52> 'const char *__single __terminated_by(0) _Nullable':'const char *__single'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     | `-DeclRefExpr {{.+}} [[var_pred_13]]
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK: |   |     | `-DeclRefExpr {{.+}} [[var_c_10]]
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK: |   |       `-DeclRefExpr {{.+}} [[var_d_10]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK: |       `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0)':'const char *__single' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_foo_9]]

char testFunc15(int pred, const char *__single _Nullable __null_terminated c, const char * _Nullable d) {
    const char *__null_terminated foo = pred ? c : d;
    return *foo;
}

// CHECK: |-FunctionDecl [[func_testFunc15:0x[^ ]+]] {{.+}} testFunc15
// CHECK: | |-ParmVarDecl [[var_pred_14:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_c_11:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_d_11:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_foo_10:0x[^ ]+]]
// CHECK: |   |   `-ConditionalOperator {{.+}} <col:41, col:52> 'const char *__single __terminated_by(0) _Nullable':'const char *__single'
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |     | `-DeclRefExpr {{.+}} [[var_pred_14]]
// CHECK: |   |     |-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK: |   |     | `-DeclRefExpr {{.+}} [[var_c_11]]
// CHECK: |   |     `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK: |   |       `-DeclRefExpr {{.+}} [[var_d_11]]
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK: |       `-UnaryOperator {{.+}} cannot overflow
// CHECK: |         `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0)':'const char *__single' <LValueToRValue>
// CHECK: |           `-DeclRefExpr {{.+}} [[var_foo_10]]

char testFunc16(int pred, const char * _Nullable __null_terminated c, const char * _Nullable d) {
    const char *__null_terminated foo = pred ? c : d;
    return *foo;
}

// CHECK: -FunctionDecl [[func_testFunc16:0x[^ ]+]] {{.+}} testFunc16
// CHECK:   |-ParmVarDecl [[var_pred_15:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_c_12:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_d_12:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_foo_11:0x[^ ]+]]
// CHECK:     |   `-ConditionalOperator {{.+}} <col:41, col:52> 'const char *__single __terminated_by(0) _Nullable':'const char *__single'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_pred_15]]
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_c_12]]
// CHECK:     |     `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0) _Nullable':'const char *__single' <LValueToRValue>
// CHECK:     |       `-DeclRefExpr {{.+}} [[var_d_12]]
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK:         `-UnaryOperator {{.+}} cannot overflow
// CHECK:           `-ImplicitCastExpr {{.+}} 'const char *__single __terminated_by(0)':'const char *__single' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_foo_11]]

