

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
#include <ptrcheck.h>

struct S {
    int *__counted_by(len + 1) buf;
    int len;
};

int foo(int *__counted_by(len) buf, int len) {
// CHECK-LABEL: FunctionDecl {{.+}} foo
// CHECK: {{^}}|  |-ParmVarDecl [[var_buf:0x[^ ]+]]
// CHECK: {{^}}|  |-ParmVarDecl [[var_len:0x[^ ]+]]

    struct S s = {0, -1};
// CHECK: {{^}}|    |-DeclStmt
// CHECK: {{^}}|    | `-VarDecl [[var_s:0x[^ ]+]]
// CHECK: {{^}}|    |   `-BoundsCheckExpr {{.+}}
// CHECK: {{^}}|    |     |-InitListExpr
// CHECK: {{^}}|    |     | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len + 1)':'int *__single'
// CHECK: {{^}}|    |     | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|    |     |-BinaryOperator {{.+}} 'int' '=='
// CHECK: {{^}}|    |     | |-BinaryOperator {{.+}} 'int' '+'
// CHECK: {{^}}|    |     | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK: {{^}}|    |     | | `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}|    |     | `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|    |     |-OpaqueValueExpr [[ove]]
// CHECK: {{^}}|    |     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len + 1)':'int *__single' <NullToPointer>
// CHECK: {{^}}|    |     |   `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|    |     `-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}|    |       `-UnaryOperator {{.+}} prefix '-'
// CHECK: {{^}}|    |         `-IntegerLiteral {{.+}} 1

    int *ptr = buf + 1;
// CHECK: {{^}}|    |-DeclStmt
// CHECK: {{^}}|    | `-VarDecl [[var_ptr:0x[^ ]+]]
// CHECK: {{^}}|    |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}|    |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|    |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|    |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|    |     | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|    |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|    |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|    |     | | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|    |     | | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|    |     | | |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}|    |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: {{^}}|    |     | | |   `-DeclRefExpr {{.+}} [[var_buf]]
// CHECK: {{^}}|    |     | | `-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}|    |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|    |     | |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: {{^}}|    |     | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|    |     | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}|    |     `-IntegerLiteral {{.+}} 1

    ptr = s.buf + 2;
// CHECK: {{^}}|    |-BinaryOperator {{.+}} 'int *__bidi_indexable' '='
// CHECK: {{^}}|    | |-DeclRefExpr {{.+}} [[var_ptr]]
// CHECK: {{^}}|    | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}|    |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|    |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|    |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|    |   | | | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len + 1)':'int *__single'
// CHECK: {{^}}|    |   | | | |     `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} lvalue
// CHECK: {{^}}|    |   | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|    |   | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|    |   | | | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len + 1)':'int *__single'
// CHECK: {{^}}|    |   | | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|    |   | | | |     |-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|    |   | | |-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}|    |   | | | `-DeclRefExpr {{.+}} [[var_s]]
// CHECK: {{^}}|    |   | | |-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}|    |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|    |   | | |   `-MemberExpr {{.+}} .len
// CHECK: {{^}}|    |   | | |     `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: {{^}}|    |   | | |-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}|    |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len + 1)':'int *__single' <LValueToRValue>
// CHECK: {{^}}|    |   | | |   `-MemberExpr {{.+}} .buf
// CHECK: {{^}}|    |   | | |     `-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: {{^}}|    |   | | `-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}|    |   | |   `-BinaryOperator {{.+}} 'int' '+'
// CHECK: {{^}}|    |   | |     |-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|    |   | |     `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}|    |   | |-OpaqueValueExpr [[ove_5]] {{.*}} lvalue
// CHECK: {{^}}|    |   | |-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|    |   | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single __counted_by(len + 1)':'int *__single'
// CHECK: {{^}}|    |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|    |   `-IntegerLiteral {{.+}} 2

    return *ptr;
// CHECK: {{^}}|    `-ReturnStmt
// CHECK: {{^}}|      `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|        `-UnaryOperator {{.+}} prefix '*'
// CHECK: {{^}}|          `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|            `-DeclRefExpr {{.+}} [[var_ptr]]
}

struct S_Nullable {
    int *__counted_by_or_null(len + 1) buf;
    int len;
};

int bar(int *__counted_by_or_null(len) buf, int len) {
// CHECK-LABEL: FunctionDecl {{.+}} bar
// CHECK: {{^}}   |-ParmVarDecl [[var_buf_1:0x[^ ]+]]
// CHECK: {{^}}   |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: {{^}}   | `-DependerDeclsAttr

    struct S_Nullable s = {0, -1};
// CHECK: {{^}}     |-DeclStmt
// CHECK: {{^}}     | `-VarDecl [[var_s_1:0x[^ ]+]]
// CHECK: {{^}}     |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}     |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}     |     | |-InitListExpr
// CHECK: {{^}}     |     | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(len + 1)':'int *__single'
// CHECK: {{^}}     |     | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}     |     | |-OpaqueValueExpr [[ove_8]]
// CHECK: {{^}}     |     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len + 1)':'int *__single' <NullToPointer>
// CHECK: {{^}}     |     | |   `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}     |     | `-OpaqueValueExpr [[ove_9]]
// CHECK: {{^}}     |     |   `-UnaryOperator {{.+}} prefix '-'
// CHECK: {{^}}     |     |     `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}     |     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by_or_null(len + 1)':'int *__single'
// CHECK: {{^}}     |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'

    int *ptr = buf + 1;
// CHECK: {{^}}     |-DeclStmt
// CHECK: {{^}}     | `-VarDecl [[var_ptr_1:0x[^ ]+]]
// CHECK: {{^}}     |   `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}     |     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}     |     | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}     |     | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}     |     | | | |-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}     |     | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}     |     | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}     |     | | | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}     |     | | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}     |     | | |-OpaqueValueExpr [[ove_10]]
// CHECK: {{^}}     |     | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <LValueToRValue>
// CHECK: {{^}}     |     | | |   `-DeclRefExpr {{.+}} [[var_buf_1]]
// CHECK: {{^}}     |     | | `-OpaqueValueExpr [[ove_11]]
// CHECK: {{^}}     |     | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}     |     | |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: {{^}}     |     | |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}     |     | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: {{^}}     |     `-IntegerLiteral {{.+}} 1

    ptr = s.buf + 2;
// CHECK: {{^}}     |-BinaryOperator {{.+}} 'int *__bidi_indexable' '='
// CHECK: {{^}}     | |-DeclRefExpr {{.+}} [[var_ptr_1]]
// CHECK: {{^}}     | `-BinaryOperator {{.+}} 'int *__bidi_indexable' '+'
// CHECK: {{^}}     |   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}     |   | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}     |   | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}     |   | | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(len + 1)':'int *__single'
// CHECK: {{^}}     |   | | | |     `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} lvalue
// CHECK: {{^}}     |   | | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}     |   | | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}     |   | | | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __counted_by_or_null(len + 1)':'int *__single'
// CHECK: {{^}}     |   | | | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}     |   | | | |     |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}     |   | | |-OpaqueValueExpr [[ove_13]]
// CHECK: {{^}}     |   | | | `-DeclRefExpr {{.+}} [[var_s_1]]
// CHECK: {{^}}     |   | | |-OpaqueValueExpr [[ove_15]]
// CHECK: {{^}}     |   | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}     |   | | |   `-MemberExpr {{.+}} .len
// CHECK: {{^}}     |   | | |     `-OpaqueValueExpr [[ove_13]] {{.*}} lvalue
// CHECK: {{^}}     |   | | |-OpaqueValueExpr [[ove_12]]
// CHECK: {{^}}     |   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len + 1)':'int *__single' <LValueToRValue>
// CHECK: {{^}}     |   | | |   `-MemberExpr {{.+}} .buf
// CHECK: {{^}}     |   | | |     `-OpaqueValueExpr [[ove_13]] {{.*}} lvalue
// CHECK: {{^}}     |   | | `-OpaqueValueExpr [[ove_14]]
// CHECK: {{^}}     |   | |   `-BinaryOperator {{.+}} 'int' '+'
// CHECK: {{^}}     |   | |     |-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: {{^}}     |   | |     `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}     |   | |-OpaqueValueExpr [[ove_13]] {{.*}} lvalue
// CHECK: {{^}}     |   | |-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: {{^}}     |   | |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single __counted_by_or_null(len + 1)':'int *__single'
// CHECK: {{^}}     |   | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: {{^}}     |   `-IntegerLiteral {{.+}} 2

    return *ptr;
// CHECK: {{^}}     `-ReturnStmt
// CHECK: {{^}}       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}         `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}           `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}             `-DeclRefExpr {{.+}} [[var_ptr_1]]
}
