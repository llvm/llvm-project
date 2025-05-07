

// RUN: %clang_cc1 -fbounds-safety -ast-dump %s 2>&1 | FileCheck %s
// RN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Inner {
    int * __counted_by(len) ptr;
    int len;
};
struct Outer {
    struct Inner hdr;
    int fam[__counted_by(hdr.len)];
};

struct Outer * __sized_by(sizeof(struct Outer) + sizeof(int) * len) bar(int len);
int * __counted_by(len) baz(int len);

// CHECK: |-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK: |-FunctionDecl [[func_baz:0x[^ ]+]] {{.+}} baz

struct Outer *foo(int len) {
    int * p2 = baz(len);
    struct Outer * __single p = bar(len);
    p->hdr.len = len;
    p->hdr.ptr = p2;
    return p;
}

// CHECK-LABEL: foo
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | |   `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|   |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |     | | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|   |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     | |   `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|   |     | `-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|   |     |   `-CallExpr
// CHECK-NEXT: {{^}}|   |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}|   |     |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     |-BoundsCheckExpr {{.+}} 'p2 <= __builtin_get_pointer_upper_bound(p2) && __builtin_get_pointer_lower_bound(p2) <= p2 && len <= __builtin_get_pointer_upper_bound(p2) - p2 && 0 <= len'
// CHECK-NEXT: {{^}}|   |     | |-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     | | `-PredefinedBoundsCheckExpr {{.+}} 'struct Outer *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK-NEXT: {{^}}|   |     | |   |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}|   |     | |   |   | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}|   |     | |   |   | | |   `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | |   |   | | |     |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |     | |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}|   |     | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|   |     | |   | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}|   |     | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}|   |     | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|   |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   |     |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   |     |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   |     |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   |     |   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   |     |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|   |     |   |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|   |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |     |     | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|   |     |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|   |     |     |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |     |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   |     |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|   |     |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   |     |       `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|   |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|   |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK-NEXT: {{^}}|   |     |   | | |-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}|   |     |   | | |-ImplicitCastExpr {{.+}} 'struct Outer *' <BitCast>
// CHECK-NEXT: {{^}}|   |     |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|   |     |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|   |     |   | | |   | `-ImplicitCastExpr {{.+}} 'struct Outer *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   |     |   | | |   |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}|   |     |   | | |   `-AssumptionExpr
// CHECK-NEXT: {{^}}|   |     |   | | |     |-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |     |   | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK-NEXT: {{^}}|   |     |   | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |     |   | | |       | `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |     |   | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |     |   | | |         `-IntegerLiteral {{.+}} 0
// CHECK:      {{^}}|   |     |   | |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|   |     |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     |   | |   `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|   |     |   | |-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|   |     |   | | `-CallExpr
// CHECK-NEXT: {{^}}|   |     |   | |   |-ImplicitCastExpr {{.+}} 'struct Outer *__single __sized_by(16UL + 4UL * len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|   |     |   | |   | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK-NEXT: {{^}}|   |     |   | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |   | `-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|   |     |   |   `-BinaryOperator {{.+}} 'unsigned long' '+'
// CHECK-NEXT: {{^}}|   |     |   |     |-IntegerLiteral {{.+}} 16
// CHECK-NEXT: {{^}}|   |     |   |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK-NEXT: {{^}}|   |     |   |       |-IntegerLiteral {{.+}} 4
// CHECK-NEXT: {{^}}|   |     |   |       `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}|   |     |   |         `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |   |-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:      {{^}}|   |     |   |-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}|   |     |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'unsigned long'
// CHECK:      {{^}}|   |     |-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|   |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |     |   `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|   |     `-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|   |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|   |         `-DeclRefExpr {{.+}} [[var_p2]]
// CHECK-NEXT: {{^}}|   |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}|   | |-MemberExpr {{.+}} .len
// CHECK-NEXT: {{^}}|   | | `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}|   | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | |     `-DeclRefExpr {{.+}} [[var_p]]
// CHECK-NEXT: {{^}}|   | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|   |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|   | |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}|   | | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}|   | | | `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}|   | | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|   | | |     `-DeclRefExpr {{.+}} [[var_p]]
// CHECK-NEXT: {{^}}|   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|   | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_2]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}|   | |-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK-NEXT: {{^}}|         | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}|         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}|         | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|         | | | | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}|         | | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}|         | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|         | | |   `-MemberExpr {{.+}} .len
// CHECK-NEXT: {{^}}|         | | |     `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}|         | | |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}|         | `-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|         |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|         |     `-DeclRefExpr {{.+}} [[var_p]]
// CHECK-NEXT: {{^}}|         `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct Outer *__single'

struct Outer *foo2(int len) {
    int * p2 = baz(len);
    struct Outer * __single p = bar(len);
    p->hdr.ptr = p2;
    p->hdr.len = len;
    return p;
}

// CHECK-LABEL:  foo2
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_p2_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}    |     | | |-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}    |     | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    |     | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}    |     | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     | | | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}    |     | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:      {{^}}    |     | |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}    |     | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |     | |   `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}    |     | `-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}    |     |   `-CallExpr
// CHECK-NEXT: {{^}}    |     |     |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    |     |     | `-DeclRefExpr {{.+}} [[func_baz]]
// CHECK-NEXT: {{^}}    |     |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:      {{^}}    |     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:      {{^}}    |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK:      {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    |     |-BoundsCheckExpr {{.+}} 'p2 <= __builtin_get_pointer_upper_bound(p2) && __builtin_get_pointer_lower_bound(p2) <= p2 && len <= __builtin_get_pointer_upper_bound(p2) - p2 && 0 <= len'
// CHECK-NEXT: {{^}}    |     | |-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     | | `-PredefinedBoundsCheckExpr {{.+}} 'struct Outer *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK-NEXT: {{^}}    |     | |   |-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}    |     | |   |   | | |-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}    |     | |   |   | | |   `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    |     | |   |   | | |     |-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'unsigned long'
// CHECK:      {{^}}    |     | |   |-OpaqueValueExpr [[ove_11]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}    |     | |   |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}    |     | |   | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}    |     | |   |   `-OpaqueValueExpr [[ove_11]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}    |     | |   `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}    |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    |     |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    |     |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    |     |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     |   | | | `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |   | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    |     |   | |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    |     |   |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}    |     |   |   | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}    |     |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    |     |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    |     |     | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}    |     |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}    |     |     |   |-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}    |     |     |   | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |     |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     |     |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |     |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}    |     |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}    |     |       `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}    |     |-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}    |     | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    |     |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    |     |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK-NEXT: {{^}}    |     |   | | |-OpaqueValueExpr [[ove_12]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}    |     |   | | |-ImplicitCastExpr {{.+}} 'struct Outer *' <BitCast>
// CHECK-NEXT: {{^}}    |     |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}    |     |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}    |     |   | | |   | `-ImplicitCastExpr {{.+}} 'struct Outer *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |     |   | | |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}    |     |   | | |   `-AssumptionExpr
// CHECK-NEXT: {{^}}    |     |   | | |     |-OpaqueValueExpr [[ove_14]] {{.*}} 'unsigned long'
// CHECK:      {{^}}    |     |   | | |     `-BinaryOperator {{.+}} 'int' '>='
// CHECK-NEXT: {{^}}    |     |   | | |       |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    |     |   | | |       | `-OpaqueValueExpr [[ove_14]] {{.*}} 'unsigned long'
// CHECK:      {{^}}    |     |   | | |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}    |     |   | | |         `-IntegerLiteral {{.+}} 0
// CHECK:      {{^}}    |     |   | |-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}    |     |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |     |   | |   `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}    |     |   | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}    |     |   | | `-CallExpr
// CHECK-NEXT: {{^}}    |     |   | |   |-ImplicitCastExpr {{.+}} 'struct Outer *__single __sized_by(16UL + 4UL * len)(*__single)(int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    |     |   | |   | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK-NEXT: {{^}}    |     |   | |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    |     |   | `-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}    |     |   |   `-BinaryOperator {{.+}} 'unsigned long' '+'
// CHECK-NEXT: {{^}}    |     |   |     |-IntegerLiteral {{.+}} 16
// CHECK-NEXT: {{^}}    |     |   |     `-BinaryOperator {{.+}} 'unsigned long' '*'
// CHECK-NEXT: {{^}}    |     |   |       |-IntegerLiteral {{.+}} 4
// CHECK-NEXT: {{^}}    |     |   |       `-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK-NEXT: {{^}}    |     |   |         `-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    |     |   |-OpaqueValueExpr [[ove_13]] {{.*}} 'int'
// CHECK:      {{^}}    |     |   |-OpaqueValueExpr [[ove_12]] {{.*}} 'struct Outer *__single __sized_by(16UL + 4UL * len)':'struct Outer *__single'
// CHECK:      {{^}}    |     |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'unsigned long'
// CHECK:      {{^}}    |     |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}    |     | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}    |     |   `-DeclRefExpr {{.+}} [[var_p2_1]]
// CHECK-NEXT: {{^}}    |     `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}    |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}    |         `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}    |-BinaryOperator {{.+}} 'int *__single __counted_by(len)':'int *__single' '='
// CHECK-NEXT: {{^}}    | |-MemberExpr {{.+}} .ptr
// CHECK-NEXT: {{^}}    | | `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}    | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}    | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-BinaryOperator {{.+}} 'int' '='
// CHECK-NEXT: {{^}}    | | |-MemberExpr {{.+}} .len
// CHECK-NEXT: {{^}}    | | | `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}    | | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}    | | |     `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_11]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:      {{^}}    | |-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}          |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}          | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK-NEXT: {{^}}          | | |-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}          | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK-NEXT: {{^}}          | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}          | | | | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}          | | | |   `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}          | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}          | | |   `-MemberExpr {{.+}} .len
// CHECK-NEXT: {{^}}          | | |     `-MemberExpr {{.+}} ->hdr
// CHECK-NEXT: {{^}}          | | |       `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct Outer *__single'
// CHECK:      {{^}}          | `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}          |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}          |     `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}          `-OpaqueValueExpr [[ove_17]] {{.*}} 'struct Outer *__single'

