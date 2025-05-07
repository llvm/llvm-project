

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -emit-llvm %s -o /dev/null

#include <ptrcheck.h>

struct Packet {
    int *__counted_by(len) buf;
    int len;
};

void *__sized_by(siz) my_alloc(int siz);
int *__sized_by(cnt) my_alloc_int(int cnt);

// CHECK: |-FunctionDecl [[func_my_alloc:0x[^ ]+]] {{.+}} my_alloc
// CHECK: |-FunctionDecl [[func_my_alloc_int:0x[^ ]+]] {{.+}} my_alloc_int

void Foo(void) {
    struct Packet p;
    int siz = 10 * sizeof(int);
    p.buf = my_alloc_int(siz);
    p.len = 10;
}
// CHECK-LABEL: |-FunctionDecl {{.+}} Foo
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_p:0x[^ ]+]]
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_siz_1:0x[^ ]+]]
// CHECK: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}|   |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}|   |       | `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}|   |       `-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|   | |-BoundsCheckExpr
// CHECK: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK: {{^}}|   | | | |-MemberExpr {{.+}} .buf
// CHECK: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | | |       | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int *__single __sized_by(cnt)':'int *__single'
// CHECK: {{^}}|   | | |       | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |     |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|   | |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}|   | |-OpaqueValueExpr [[ove]]
// CHECK: {{^}}|   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __sized_by(cnt)':'int *__single'
// CHECK: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK: {{^}}|   | |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}|   | |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}|   | |   | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |   | | |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __sized_by(cnt)':'int *__single'
// CHECK: {{^}}|   | |   | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}|   | |   | |-OpaqueValueExpr [[ove_2]]
// CHECK: {{^}}|   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|   | |   | |   `-DeclRefExpr {{.+}} [[var_siz_1]]
// CHECK: {{^}}|   | |   | `-OpaqueValueExpr [[ove_1]]
// CHECK: {{^}}|   | |   |   `-CallExpr
// CHECK: {{^}}|   | |   |     |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(cnt)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}|   | |   |     | `-DeclRefExpr {{.+}} [[func_my_alloc_int]]
// CHECK: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}|   | |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: {{^}}|   | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'int *__single __sized_by(cnt)':'int *__single'
// CHECK: {{^}}|   | `-OpaqueValueExpr [[ove_3]]
// CHECK: {{^}}|   |   `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK: {{^}}|     | |-MemberExpr {{.+}} .len
// CHECK: {{^}}|     | | `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'


void FooBitCast(void) {
    struct Packet p;
    int siz = 10 * sizeof(int);
    p.buf = my_alloc(siz);
    p.len = 10;
}

// CHECK-LABEL: |-FunctionDecl {{.+}} FooBitCast
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_siz_2:0x[^ ]+]]
// CHECK: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}|   |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: {{^}}|   |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}|   |       | `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}|   |       `-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|   | |-BoundsCheckExpr
// CHECK: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK: {{^}}|   | | | |-MemberExpr {{.+}} .buf
// CHECK: {{^}}|   | | | | `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | | |         | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}|   | | |         | | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|   | |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|   | |-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: {{^}}|   | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|   | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|   | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}|   | |     | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}|   | |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}|   | |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}|   | |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}|   | |     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|   | |     | | |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}|   | |     | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|   | |     | |-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}|   | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|   | |     | |   `-DeclRefExpr {{.+}} [[var_siz_2]]
// CHECK: {{^}}|   | |     | `-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}|   | |     |   `-CallExpr
// CHECK: {{^}}|   | |     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(siz)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}|   | |     |     | `-DeclRefExpr {{.+}} [[func_my_alloc]]
// CHECK: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|   | |     |-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|   | |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}|   | `-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}|   |   `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK: {{^}}|     | |-MemberExpr {{.+}} .len
// CHECK: {{^}}|     | | `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
void FooCCast(void) {
    struct Packet p;
    int siz = 10 * sizeof(int);
    p.buf = (int*)my_alloc(siz);
    p.len = 10;
}

// CHECK-LABEL: `-FunctionDecl {{.+}} FooCCast
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_p_2:0x[^ ]+]]
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_siz_3:0x[^ ]+]]
// CHECK: {{^}}    |   `-ImplicitCastExpr {{.+}} 'int' <IntegralCast>
// CHECK: {{^}}    |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK: {{^}}    |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: {{^}}    |       | `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}    |       `-UnaryExprOrTypeTraitExpr
// CHECK: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    | |-BoundsCheckExpr
// CHECK: {{^}}    | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK: {{^}}    | | | |-MemberExpr {{.+}} .buf
// CHECK: {{^}}    | | | | `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | | |   `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | | |         | | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}    | | |         | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    | |   | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}    | |   |   | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    | |     | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}    | |     |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    | |     |   | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |     |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |       |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}    | |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: {{^}}    | |-OpaqueValueExpr [[ove_8]]
// CHECK: {{^}}    | | `-CStyleCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: {{^}}    | |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    | |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    | |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: {{^}}    | |     | | |-OpaqueValueExpr [[ove_9]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}    | |     | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: {{^}}    | |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: {{^}}    | |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: {{^}}    | |     | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |     | | |   |   `-OpaqueValueExpr [[ove_9]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}    | |     | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: {{^}}    | |     | |-OpaqueValueExpr [[ove_10]]
// CHECK: {{^}}    | |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    | |     | |   `-DeclRefExpr {{.+}} [[var_siz_3]]
// CHECK: {{^}}    | |     | `-OpaqueValueExpr [[ove_9]]
// CHECK: {{^}}    | |     |   `-CallExpr
// CHECK: {{^}}    | |     |     |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(siz)(*__single)(int)' <FunctionToPointerDecay>
// CHECK: {{^}}    | |     |     | `-DeclRefExpr {{.+}} [[func_my_alloc]]
// CHECK: {{^}}    | |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: {{^}}    | |     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK: {{^}}    | |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'void *__single __sized_by(siz)':'void *__single'
// CHECK: {{^}}    | `-OpaqueValueExpr [[ove_11]]
// CHECK: {{^}}    |   `-IntegerLiteral {{.+}} 10
// CHECK: {{^}}    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}      |-BinaryOperator {{.+}} 'int' '='
// CHECK: {{^}}      | |-MemberExpr {{.+}} .len
// CHECK: {{^}}      | | `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK: {{^}}      | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: {{^}}      |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}      `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
