

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Simple {
    int len;
    int fam[__counted_by(len)];
};

// rdar://132731845 the flexible arrays are not bounds checked

void simple_no_flexbase_update(struct Simple * __bidi_indexable p) {
    p->len = 11;
}
// CHECK: {{^}}|-FunctionDecl [[func_simple_no_flexbase_update:0x[^ ]+]] {{.+}} simple_no_flexbase_update
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | | `-ImplicitCastExpr {{.+}} 'struct Simple *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | | |   `-DeclRefExpr {{.+}} [[var_p]]
// CHECK-NEXT: {{^}}|     | | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|       `-IntegerLiteral {{.+}} 11

// rdar://132731845
void simple_flexbase_update(struct Simple * __bidi_indexable p) {
    struct Simple * __bidi_indexable p2 = p;
    p2->len = 11;
}
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_simple_flexbase_update:0x[^ ]+]] {{.+}} simple_flexbase_update
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'struct Simple *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | | `-ImplicitCastExpr {{.+}} 'struct Simple *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | | |   `-DeclRefExpr {{.+}} [[var_p2]]
// CHECK-NEXT: {{^}}|     | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|       `-IntegerLiteral {{.+}} 11

// rdar://132731845
void simple_flexbase_self_assign(struct Simple * __bidi_indexable p) {
    p = p;
    p->len = 11;
}
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_simple_flexbase_self_assign:0x[^ ]+]] {{.+}} simple_flexbase_self_assign
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-BinaryOperator {{.+}} 'struct Simple *__bidi_indexable' '='
// CHECK-NEXT: {{^}}|   | |-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK-NEXT: {{^}}|   | `-ImplicitCastExpr {{.+}} 'struct Simple *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | | `-ImplicitCastExpr {{.+}} 'struct Simple *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | | |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK-NEXT: {{^}}|     | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|       `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|-RecordDecl
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| |-FieldDecl
// CHECK-NEXT: {{^}}| `-FieldDecl

struct Shared {
    int len;
    int * __counted_by(len) ptr;
    int fam[__counted_by(len)];
};
int * __counted_by(len) baz(int len);
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_baz:0x[^ ]+]] {{.+}} baz
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_len:0x[^ ]+]]

void shared_no_flexbase_update(struct Shared * __bidi_indexable p) {
    int * p2 = baz(11);
    p->len = 11;
    p->ptr = p2;
}
// CHECK-NEXT: {{^}}|-FunctionDecl [[func_shared_no_flexbase_update:0x[^ ]+]] {{.+}} shared_no_flexbase_update
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p2 <= __builtin_get_pointer_upper_bound(p2) && __builtin_get_pointer_lower_bound(p2) <= p2 && 11 <= __builtin_get_pointer_upper_bound(p2) - p2 && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p2_1]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__bidi_indexable'

void shared_no_flexbase_update_reverse(struct Shared * __bidi_indexable p) {
    p->ptr = baz(11);
    p->len = 11;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_no_flexbase_update_reverse:0x[^ ]+]] {{.+}} shared_no_flexbase_update_reverse
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'baz(11) <= __builtin_get_pointer_upper_bound(baz(11)) && __builtin_get_pointer_lower_bound(baz(11)) <= baz(11) && 11 <= __builtin_get_pointer_upper_bound(baz(11)) - baz(11) && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |       | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | | |       | | |   `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   | |   | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   | |-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|   | |   | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   | |   | `-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|   | |   |   `-CallExpr
// CHECK-NEXT: {{^}}|   | |   |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   | |   |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   |-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'

void shared_flexbase_update(struct Shared * __bidi_indexable p) {
    int * p3 = baz(11);
    struct Shared * __bidi_indexable p2 = p;
    p2->ptr = p3;
    p2->len = 11;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_update:0x[^ ]+]] {{.+}} shared_flexbase_update
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_12]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p3 <= __builtin_get_pointer_upper_bound(p3) && __builtin_get_pointer_lower_bound(p3) <= p3 && 11 <= __builtin_get_pointer_upper_bound(p3) - p3 && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p2_2]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |   `-DeclRefExpr {{.+}} [[var_p3]]
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p2_2]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'

void shared_flexbase_update_reverse(struct Shared * __bidi_indexable p) {
    int * p3 = baz(11);
    struct Shared * __bidi_indexable p2 = p;
    p2->len = 11;
    p2->ptr = p3;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_update_reverse:0x[^ ]+]] {{.+}} shared_flexbase_update_reverse
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p3_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_16]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p_6]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p3 <= __builtin_get_pointer_upper_bound(p3) && __builtin_get_pointer_lower_bound(p3) <= p3 && 11 <= __builtin_get_pointer_upper_bound(p3) - p3 && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p2_3]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p3_1]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p2_3]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__bidi_indexable'

void shared_flexbase_self_assign(struct Shared * __bidi_indexable p) {
    int * p2 = baz(11);
    p = p;
    p->ptr = p2;
    p->len = 11;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_self_assign:0x[^ ]+]] {{.+}} shared_flexbase_self_assign
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-BinaryOperator {{.+}} 'struct Shared *__bidi_indexable' '='
// CHECK-NEXT: {{^}}|   | |-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK-NEXT: {{^}}|   | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p2 <= __builtin_get_pointer_upper_bound(p2) && __builtin_get_pointer_lower_bound(p2) <= p2 && 11 <= __builtin_get_pointer_upper_bound(p2) - p2 && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_21]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |   `-DeclRefExpr {{.+}} [[var_p2_4]]
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_22]] {{.*}} 'int'

void shared_flexbase_self_assign_reverse(struct Shared * __bidi_indexable p) {
    int * p2 = baz(11);
    p = p;
    p->len = 11;
    p->ptr = p2;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_self_assign_reverse:0x[^ ]+]] {{.+}} shared_flexbase_self_assign_reverse
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_8:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}|   |     | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove_23]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-BinaryOperator {{.+}} 'struct Shared *__bidi_indexable' '='
// CHECK-NEXT: {{^}}|   | |-DeclRefExpr {{.+}} [[var_p_8]]
// CHECK-NEXT: {{^}}|   | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   `-DeclRefExpr {{.+}} [[var_p_8]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p2 <= __builtin_get_pointer_upper_bound(p2) && __builtin_get_pointer_lower_bound(p2) <= p2 && 11 <= __builtin_get_pointer_upper_bound(p2) - p2 && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_8]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_25]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_25]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_25]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}|   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     `-DeclRefExpr {{.+}} [[var_p2_5]]
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_8]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_25]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_26]] {{.*}} 'int *__bidi_indexable'

void shared_flexbase_self_assign_fr(struct Shared * __bidi_indexable p) {
    p = p;
    p->ptr = p->ptr;
    p->len = 11;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_self_assign_fr:0x[^ ]+]] {{.+}} shared_flexbase_self_assign_fr
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_9:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-BinaryOperator {{.+}} 'struct Shared *__bidi_indexable' '='
// CHECK-NEXT: {{^}}|   | |-DeclRefExpr {{.+}} [[var_p_9]]
// CHECK-NEXT: {{^}}|   | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   `-DeclRefExpr {{.+}} [[var_p_9]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p->ptr <= __builtin_get_pointer_upper_bound(p->ptr) && __builtin_get_pointer_lower_bound(p->ptr) <= p->ptr && 11 <= __builtin_get_pointer_upper_bound(p->ptr) - p->ptr && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_9]]
// CHECK-NEXT: {{^}}|   | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | | |   `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | | |       | | |-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | | |       | | |     `-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   | | |       | | | `-OpaqueValueExpr [[ove_30:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_31:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_27]]
// CHECK-NEXT: {{^}}|   | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   | |   | | |-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_30]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   | |-OpaqueValueExpr [[ove_29]]
// CHECK-NEXT: {{^}}|   | |   | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |   | |   `-DeclRefExpr {{.+}} [[var_p_9]]
// CHECK-NEXT: {{^}}|   | |   | |-OpaqueValueExpr [[ove_30]]
// CHECK-NEXT: {{^}}|   | |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |   | |   `-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   | |   | |     `-OpaqueValueExpr [[ove_29]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-OpaqueValueExpr [[ove_28]]
// CHECK-NEXT: {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |   |     `-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   | |   |       `-OpaqueValueExpr [[ove_29]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   | |   |-OpaqueValueExpr [[ove_29]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   | |   |-OpaqueValueExpr [[ove_30]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_31]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_9]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_27]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_31]] {{.*}} 'int'

void shared_flexbase_self_assign_fr_reverse(struct Shared * __bidi_indexable p) {
    p = p;
    p->len = 11;
    p->ptr = p->ptr;
}
// CHECK:      {{^}}|-FunctionDecl [[func_shared_flexbase_self_assign_fr_reverse:0x[^ ]+]] {{.+}} shared_flexbase_self_assign_fr_reverse
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_10:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-BinaryOperator {{.+}} 'struct Shared *__bidi_indexable' '='
// CHECK-NEXT: {{^}}|   | |-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK-NEXT: {{^}}|   | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |   `-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK-NEXT: {{^}}|   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   | |-BoundsCheckExpr {{.+}} 'p->ptr <= __builtin_get_pointer_upper_bound(p->ptr) && __builtin_get_pointer_lower_bound(p->ptr) <= p->ptr && 11 <= __builtin_get_pointer_upper_bound(p->ptr) - p->ptr && 0 <= 11'
// CHECK-NEXT: {{^}}|   | | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   | | | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | | |   `-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK-NEXT: {{^}}|   | | | `-OpaqueValueExpr [[ove_32:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   | | | `-OpaqueValueExpr [[ove_33:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | |     | | |-OpaqueValueExpr [[ove_34:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   | |   | | |     | | |     `-OpaqueValueExpr [[ove_35:0x[^ ]+]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   | |   | | |     | | | `-OpaqueValueExpr [[ove_36:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   | |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |   | |   `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   | |   |   | `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   |     `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   | |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   | |     | | `-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
// CHECK:      {{^}}|   | |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   | |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   | |     |   | `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |     |     `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   | |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   | |       `-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_32]]
// CHECK-NEXT: {{^}}|   | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_33]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove_34]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove_34]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_36]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_35]]
// CHECK-NEXT: {{^}}|   |     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     | |   `-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK-NEXT: {{^}}|   |     | |-OpaqueValueExpr [[ove_36]]
// CHECK-NEXT: {{^}}|   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     | |   `-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|   |     | |     `-OpaqueValueExpr [[ove_35]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   |     | `-OpaqueValueExpr [[ove_34]]
// CHECK-NEXT: {{^}}|   |     |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     |     `-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|   |     |       `-OpaqueValueExpr [[ove_35]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_35]] {{.*}} 'struct Shared *__bidi_indexable'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_36]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove_34]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|     | |-MemberExpr {{.+}} ->ptr
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'struct Shared *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK-NEXT: {{^}}|     | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     |   `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     |-OpaqueValueExpr [[ove_32]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_33]] {{.*}} 'int *__bidi_indexable'

struct Double {
    int len;
    int len2;
    int fam[__counted_by(len + len2)];
};

void double_no_flexbase_update_once(struct Double * __bidi_indexable p) {
    p->len = 11;
}
// CHECK: {{^}}|-FunctionDecl [[func_double_no_flexbase_update_once:0x[^ ]+]] {{.+}} double_no_flexbase_update_once
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_11:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|     | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}|     | | | `-ImplicitCastExpr {{.+}} 'struct Double *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | | |   `-DeclRefExpr {{.+}} [[var_p_11]]
// CHECK-NEXT: {{^}}|     | | `-OpaqueValueExpr [[ove_37:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_37]] {{.*}} 'int'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_37]]
// CHECK-NEXT: {{^}}|       `-IntegerLiteral {{.+}} 11

void double_no_flexbase_update_both(struct Double * __bidi_indexable p) {
    p->len = 11;
    p->len2 = 11;
}
// CHECK-NEXT: {{^}}`-FunctionDecl [[func_double_no_flexbase_update_both:0x[^ ]+]] {{.+}} double_no_flexbase_update_both
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_p_12:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}    | | |-MemberExpr {{.+}} ->len
// CHECK-NEXT: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'struct Double *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | |   `-DeclRefExpr {{.+}} [[var_p_12]]
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_38:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_38]]
// CHECK-NEXT: {{^}}    | | `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}    | `-OpaqueValueExpr [[ove_39:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |   `-IntegerLiteral {{.+}} 11
// CHECK-NEXT: {{^}}    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}      |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}      | |-MemberExpr {{.+}} ->len2
// CHECK-NEXT: {{^}}      | | `-ImplicitCastExpr {{.+}} 'struct Double *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}      | |   `-DeclRefExpr {{.+}} [[var_p_12]]
// CHECK-NEXT: {{^}}      | `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
// CHECK:      {{^}}      |-OpaqueValueExpr [[ove_38]] {{.*}} 'int'
// CHECK:      {{^}}      `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
