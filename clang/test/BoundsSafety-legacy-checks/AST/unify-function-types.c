
// RUN: %clang_cc1 -ast-dump -fbounds-safety -verify=c %s | FileCheck %s
// RUN: %clang_cc1 -x c++ -ast-dump -fbounds-safety -fexperimental-bounds-safety-cxx -verify=cpp %s
#include <ptrcheck.h>

char funcA(char buf[__counted_by(len)], int len, int len2);
char funcB(char arr[__counted_by(size)], int size, int size2);
char funcC(char arr[__counted_by(size2)], int size, int size2);

// CHECK: TranslationUnitDecl
// CHECK: |-FunctionDecl [[func_funcA:0x[^ ]+]] {{.+}} funcA
// CHECK: | |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_len2:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_funcB:0x[^ ]+]] {{.+}} funcB
// CHECK: | |-ParmVarDecl [[var_arr:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_size2:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_funcC:0x[^ ]+]] {{.+}} funcC
// CHECK: | |-ParmVarDecl [[var_arr_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_1:0x[^ ]+]]
// CHECK: | `-ParmVarDecl [[var_size2_1:0x[^ ]+]]
// CHECK: |   `-DependerDeclsAttr

char test1(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    return (src_len % 2 == 0 ? funcA : funcB)(src_buf, src_len, src_len2);
}

// CHECK: |-FunctionDecl [[func_test1:0x[^ ]+]] {{.+}} test1
// CHECK: | |-ParmVarDecl [[var_src_buf:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_src_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_src_len2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |-BoundsCheckExpr
// CHECK: |       | | |-CallExpr
// CHECK: |       | | | |-ParenExpr
// CHECK: |       | | | | `-ConditionalOperator {{.+}} <col:13, col:40> 'char (*__single)(char *__single __counted_by(len), int, int)'
// CHECK: |       | | | |   |-BinaryOperator {{.+}} 'int' '=='
// CHECK: |       | | | |   | |-BinaryOperator {{.+}} 'int' '%'
// CHECK: |       | | | |   | | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | | | |   | | | `-DeclRefExpr {{.+}} [[var_src_len]]
// CHECK: |       | | | |   | | `-IntegerLiteral {{.+}} 2
// CHECK: |       | | | |   | `-IntegerLiteral {{.+}} 0
// CHECK: |       | | | |   |-ImplicitCastExpr {{.+}} 'char (*__single)(char *__single __counted_by(len), int, int)' <FunctionToPointerDecay>
// CHECK: |       | | | |   | `-DeclRefExpr {{.+}} [[func_funcA]]
// CHECK: |       | | | |   `-ImplicitCastExpr {{.+}} 'char (*__single)(char *__single __counted_by(size), int, int)' <FunctionToPointerDecay>
// CHECK: |       | | | |     `-DeclRefExpr {{.+}} [[func_funcB]]
// CHECK: |       | | | |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |       | | | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | | | |     | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |       | | | |     | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK: |       | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: |       | |   | |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |   |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: |       | |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |       | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |       | |     | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |       | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |       | |     |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |   | `-GetBoundExpr {{.+}} upper
// CHECK: |       | |     |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |       | |       |-IntegerLiteral {{.+}} 0
// CHECK: |       | |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |       | |-OpaqueValueExpr [[ove]]
// CHECK: |       | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |       | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |       | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |       | |   | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |       | |   | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |       | |   | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |       | |   | | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |       | |   | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |       | |   | |-OpaqueValueExpr [[ove_1]]
// CHECK: |       | |   | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(src_len)':'char *__single' <LValueToRValue>
// CHECK: |       | |   | |   `-DeclRefExpr {{.+}} [[var_src_buf]]
// CHECK: |       | |   | `-OpaqueValueExpr [[ove_2]]
// CHECK: |       | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |   |     `-DeclRefExpr {{.+}} [[var_src_len]]
// CHECK: |       | |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |       | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |       | |-OpaqueValueExpr [[ove_3]]
// CHECK: |       | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       | |   `-DeclRefExpr {{.+}} [[var_src_len]]
// CHECK: |       | `-OpaqueValueExpr [[ove_4]]
// CHECK: |       |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |       |     `-DeclRefExpr {{.+}} [[var_src_len2]]
// CHECK: |       |-OpaqueValueExpr [[ove]] {{.*}} 'char *__bidi_indexable'
// CHECK: |       |-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'


char test2(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    // c-error@+2{{conditional expression evaluates functions with incompatible bound attributes 'char (*__single)(char *__single __counted_by(len), int, int)' (aka 'char (*__single)(char *__single, int, int)') and 'char (*__single)(char *__single __counted_by(size2), int, int)' (aka 'char (*__single)(char *__single, int, int)')}}
    // cpp-error@+1{{conditional expression evaluates functions with incompatible bound attributes 'char (char *__single __counted_by(len), int, int)' (aka 'char (char *__single, int, int)') and 'char (char *__single __counted_by(size2), int, int)' (aka 'char (char *__single, int, int)')}}
    return (src_len % 2 == 0 ? funcA : funcC)(src_buf, src_len, src_len2);
}

char * __counted_by(len) funcD(char buf[__counted_by(len)], int len, int len2);
char * __counted_by(size) funcE(char arr[__counted_by(size)], int size, int size2);
char * __counted_by(size2) funcF(char arr[__counted_by(size)], int size, int size2);

// CHECK: |-FunctionDecl [[func_funcD:0x[^ ]+]] {{.+}} funcD
// CHECK: | |-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_len2_1:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_funcE:0x[^ ]+]] {{.+}} funcE
// CHECK: | |-ParmVarDecl [[var_arr_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_2:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_size2_2:0x[^ ]+]]
// CHECK: |-FunctionDecl [[func_funcF:0x[^ ]+]] {{.+}} funcF
// CHECK: | |-ParmVarDecl [[var_arr_3:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_size_3:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | `-ParmVarDecl [[var_size2_3:0x[^ ]+]]

char * __counted_by(src_len) test3(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    return (src_len % 2 == 0 ? funcD : funcE)(src_buf, src_len, src_len2);
}

// CHECK: |-FunctionDecl [[func_test3:0x[^ ]+]] {{.+}} test3
// CHECK: | |-ParmVarDecl [[var_src_buf_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_src_len_2:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
// CHECK: | |-ParmVarDecl [[var_src_len2_2:0x[^ ]+]]
// CHECK: | `-CompoundStmt
// CHECK: |   `-ReturnStmt
// CHECK: |     `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(src_len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: |         | | |   | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         | | |   | |     | | |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |         | | |   | |     | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | |   | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | |   | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__single __counted_by(len)':'char *__single'
// CHECK: |         | | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_6]]
// CHECK: |         | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK: |         | |   | | |-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |         | |   | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | |   | | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         | |   | | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |         | |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK: |         | |   | |-OpaqueValueExpr [[ove_7]]
// CHECK: |         | |   | | `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(src_len)':'char *__single' <LValueToRValue>
// CHECK: |         | |   | |   `-DeclRefExpr {{.+}} [[var_src_buf_2]]
// CHECK: |         | |   | `-OpaqueValueExpr [[ove_8]]
// CHECK: |         | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   |     `-DeclRefExpr {{.+}} [[var_src_len_2]]
// CHECK: |         | |   |-OpaqueValueExpr [[ove_7]] {{.*}} 'char *__single __counted_by(src_len)':'char *__single'
// CHECK: |         | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_9]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_src_len_2]]
// CHECK: |         | |-OpaqueValueExpr [[ove_10]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_src_len2_2]]
// CHECK: |         | `-OpaqueValueExpr [[ove_5]]
// CHECK: |         |   `-BoundsCheckExpr
// CHECK: |         |     |-CallExpr
// CHECK: |         |     | |-ParenExpr
// CHECK: |         |     | | `-ConditionalOperator {{.+}} <col:13, col:40> 'char *__single __counted_by(len)(*__single)(char *__single __counted_by(len), int, int)'
// CHECK: |         |     | |   |-BinaryOperator {{.+}} 'int' '=='
// CHECK: |         |     | |   | |-BinaryOperator {{.+}} 'int' '%'
// CHECK: |         |     | |   | | |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |     | |   | | | `-DeclRefExpr {{.+}} [[var_src_len_2]]
// CHECK: |         |     | |   | | `-IntegerLiteral {{.+}} 2
// CHECK: |         |     | |   | `-IntegerLiteral {{.+}} 0
// CHECK: |         |     | |   |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)(*__single)(char *__single __counted_by(len), int, int)' <FunctionToPointerDecay>
// CHECK: |         |     | |   | `-DeclRefExpr {{.+}} [[func_funcD]]
// CHECK: |         |     | |   `-ImplicitCastExpr {{.+}} 'char *__single __counted_by(size)(*__single)(char *__single __counted_by(size), int, int)' <FunctionToPointerDecay>
// CHECK: |         |     | |     `-DeclRefExpr {{.+}} [[func_funcE]]
// CHECK: |         |     | |-ImplicitCastExpr {{.+}} 'char *__single __counted_by(len)':'char *__single' <BoundsSafetyPointerCast>
// CHECK: |         |     | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |     | |-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: |         |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |         |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         |       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         |       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |       | | |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |       | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |       | | `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |       | |   `-GetBoundExpr {{.+}} upper
// CHECK: |         |       | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |       |   | `-GetBoundExpr {{.+}} lower
// CHECK: |         |       |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |       |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |       `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         |         |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |         | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |         |         | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: |         |         | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |         |         |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |         |   | `-GetBoundExpr {{.+}} upper
// CHECK: |         |         |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |         |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |         |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |         `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |           |-IntegerLiteral {{.+}} 0
// CHECK: |         |           `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: |         |-OpaqueValueExpr [[ove_6]] {{.*}} 'char *__bidi_indexable'
// CHECK: |         |-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: |         |-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: |         `-OpaqueValueExpr [[ove_5]] {{.*}} 'char *__single __counted_by(len)':'char *__single'

char test4(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    // c-error@+2{{conditional expression evaluates functions with incompatible bound attributes 'char *__single __counted_by(len)(*__single)(char *__single __counted_by(len), int, int)' (aka 'char *__single(*__single)(char *__single, int, int)') and 'char *__single __counted_by(size2)(*__single)(char *__single __counted_by(size), int, int)' (aka 'char *__single(*__single)(char *__single, int, int)')}}
    // cpp-error@+1{{conditional expression evaluates functions with incompatible bound attributes 'char *__single __counted_by(len)(char *__single __counted_by(len), int, int)' (aka 'char *__single(char *__single, int, int)') and 'char *__single __counted_by(size2)(char *__single __counted_by(size), int, int)' (aka 'char *__single(char *__single, int, int)')}}
    return *(src_len % 2 == 0 ? funcD : funcF)(src_buf, src_len, src_len2);
}

#ifdef __cplusplus
char * __counted_by(src_len) test5(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    auto &funcRefD (funcD);
    auto &funcRefE (funcE);
    return (src_len % 2 == 0 ? funcRefD : funcRefE)(src_buf, src_len, src_len2);
}

char * __counted_by(src_len) test6(char src_buf[__counted_by(src_len)], int src_len, int src_len2) {
    auto &funcRefD (funcD);
    auto &funcRefF (funcF);
    // cpp-error@+1{{conditional expression evaluates functions with incompatible bound attributes 'char *__single __counted_by(len)(char *__single __counted_by(len), int, int)' (aka 'char *__single(char *__single, int, int)') and 'char *__single __counted_by(size2)(char *__single __counted_by(size), int, int)' (aka 'char *__single(char *__single, int, int)')}}
    return (src_len % 2 == 0 ? funcRefD : funcRefF)(src_buf, src_len, src_len2);
}

#endif
