

// RUN: %clang_cc1 -ast-dump -verify -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -x objective-c -ast-dump -verify -fbounds-safety -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: |-RecordDecl
struct unsized;

// CHECK: |-FunctionDecl [[func_unsizedSizedByToSizedBy:0x[^ ]+]] {{.+}} unsizedSizedByToSizedBy
// CHECK: | |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
void unsizedSizedByToSizedBy(struct unsized * __sized_by(len) p, int len) {
// CHECK: | `-CompoundStmt
// CHECK: |   |-DeclStmt
// CHECK: |   | `-VarDecl [[var_size:0x[^ ]+]]
// CHECK: |   |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |   |   | `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |   |   `-DependerDeclsAttr
    int size = len;
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2:0x[^ ]+]]
// CHECK: |       `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && size <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__bidi_indexable)p && 0 <= size'
// CHECK: |         |-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' <BoundsSafetyPointerCast>
// CHECK: |         | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         |     | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |         |     | | |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         | | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK: |         | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         | | | `-GetBoundExpr {{.+}} upper
// CHECK: |         | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         | |   |-GetBoundExpr {{.+}} lower
// CHECK: |         | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         | |   `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK: |         | |     `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: |         |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |   | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'long'
// CHECK: |         |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: |         |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         |   |   | `-GetBoundExpr {{.+}} upper
// CHECK: |         |   |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK: |         |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK: |         |   |       `-OpaqueValueExpr [[ove]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK: |         |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK: |         |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |         |     | `-IntegerLiteral {{.+}} 0
// CHECK: |         |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'long'
// CHECK: |         |-OpaqueValueExpr [[ove]]
// CHECK: |         | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK: |         |   | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |         |   | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK: |         |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         |   | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK: |         |   | | |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |         |   | | |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |         |   | |-OpaqueValueExpr [[ove_1]]
// CHECK: |         |   | | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single' <LValueToRValue>
// CHECK: |         |   | |   `-DeclRefExpr {{.+}} [[var_p]]
// CHECK: |         |   | `-OpaqueValueExpr [[ove_2]]
// CHECK: |         |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |   |     `-DeclRefExpr {{.+}} [[var_len]]
// CHECK: |         |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |         |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK: |         `-OpaqueValueExpr [[ove_3]]
// CHECK: |           `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: |             `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |               `-DeclRefExpr {{.+}} [[var_size]]
    struct unsized * __single __sized_by(size) p2 = p;
}

// CHECK: |-RecordDecl
struct Bar {
// CHECK: | |-FieldDecl
// CHECK: | | `-DependerDeclsAttr
    int size;
// CHECK: | `-FieldDecl
    struct unsized * __single __sized_by(size) p;
};

// CHECK: |-FunctionDecl [[func_structMemberUnsizedSizedBy:0x[^ ]+]] {{.+}} structMemberUnsizedSizedBy
// CHECK:   |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// expected-note@+2{{consider adding '__sized_by(len)' to 'p'}}
// expected-note@+1{{consider adding '__sized_by(10)' to 'p'}}
void structMemberUnsizedSizedBy(struct unsized * p, int len) {
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_bar:0x[^ ]+]]
// CHECK:     |   `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && 10 <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__single)p && 0 <= 10'
// CHECK:     |     |-InitListExpr
// CHECK:     |     | |-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int'
// CHECK:     |     | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'struct unsized *__single'
// CHECK:     |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     | | | `-GetBoundExpr {{.+}} upper
// CHECK:     |     | | |   `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     | | |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | |   |-GetBoundExpr {{.+}} lower
// CHECK:     |     | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     | |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     | |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     |   | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:     |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     |     |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     |     |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:     |     |   |   |   `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     |   |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     |   |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK:     |     |   |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'struct unsized *__single'
// CHECK:     |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |     |-IntegerLiteral {{.+}} 0
// CHECK:     |     |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int'
// CHECK:     |     |-OpaqueValueExpr [[ove_4]]
// CHECK:     |     | `-IntegerLiteral {{.+}} 10
// CHECK:     |     `-OpaqueValueExpr [[ove_5]]
// CHECK:     |       `-ImplicitCastExpr {{.+}} 'struct unsized *__single' <LValueToRValue>
// CHECK:     |         `-DeclRefExpr {{.+}} [[var_p_1]]
    struct Bar bar = { .size = 10, .p = p }; // expected-error{{initializing 'bar.p' of type 'struct unsized *__single __sized_by(size)' (aka 'struct unsized *__single') and size value of 10 with 'struct unsized *__single' and pointee of size 0 always fails}}
// CHECK:     |-BinaryOperator {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' '='
// CHECK:     | |-MemberExpr {{.+}} .p
// CHECK:     | | `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK:     | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'struct unsized *__single'
    bar.p = p; // expected-warning{{size value is not statically known: assigning to 'struct unsized *__single __sized_by(size)' (aka 'struct unsized *__single') from 'struct unsized *__single' is invalid for any size greater than 0}}
// CHECK:     |-BinaryOperator {{.+}} 'int' '='
// CHECK:     | |-MemberExpr {{.+}} .size
// CHECK:     | | `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK:     | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
    bar.size = len; // expected-note{{size assigned here}}

// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_bar2:0x[^ ]+]]
// CHECK:     |   `-BoundsCheckExpr {{.+}} 'bar.p <= __builtin_get_pointer_upper_bound(bar.p) && __builtin_get_pointer_lower_bound(bar.p) <= bar.p && len <= (char *)__builtin_get_pointer_upper_bound(bar.p) - (char *__bidi_indexable)bar.p && 0 <= len'
// CHECK:     |     |-InitListExpr
// CHECK:     |     | |-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK:     |     | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' <BoundsSafetyPointerCast>
// CHECK:     |     |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     |       | | |-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single'
// CHECK:     |     |       | | |     `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} lvalue
// CHECK:     |     |       | | |   `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int'
// CHECK:     |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:     |     | | | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     | | | `-GetBoundExpr {{.+}} upper
// CHECK:     |     | | |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | |   |-GetBoundExpr {{.+}} lower
// CHECK:     |     | |   | `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     | |   `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:     |     | |     `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     |   | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:     |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     |     |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     |     |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:     |     |   |   |   `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:     |     |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:     |     |   |       `-OpaqueValueExpr [[ove_7]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |     |-IntegerLiteral {{.+}} 0
// CHECK:     |     |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:     |     |-OpaqueValueExpr [[ove_6]]
// CHECK:     |     | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |     |   `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK:     |     `-OpaqueValueExpr [[ove_7]]
// CHECK:     |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:     |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:     |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK:     |         | | |-OpaqueValueExpr [[ove_8]] {{.*}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single'
// CHECK:     |         | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK:     |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:     |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     |         | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:     |         | | |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single'
// CHECK:     |         | | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:     |         | |-OpaqueValueExpr [[ove_9]]
// CHECK:     |         | | `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK:     |         | |-OpaqueValueExpr [[ove_10]]
// CHECK:     |         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |         | |   `-MemberExpr {{.+}} .size
// CHECK:     |         | |     `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK:     |         | `-OpaqueValueExpr [[ove_8]]
// CHECK:     |         |   `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' <LValueToRValue>
// CHECK:     |         |     `-MemberExpr {{.+}} .p
// CHECK:     |         |       `-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK:     |         |-OpaqueValueExpr [[ove_9]] {{.*}} lvalue
// CHECK:     |         |-OpaqueValueExpr [[ove_10]] {{.*}} 'int'
// CHECK:     |         `-OpaqueValueExpr [[ove_8]] {{.*}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single'
    struct Bar bar2 = { .size = len, .p = bar.p };
// CHECK:     |-BinaryOperator {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' '='
// CHECK:     | |-MemberExpr {{.+}} .p
// CHECK:     | | `-DeclRefExpr {{.+}} [[var_bar2]]
// CHECK:     | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' <BoundsSafetyPointerCast>
// CHECK:     |   `-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:     |       | | |-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single'
// CHECK:     |       | | |     `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} lvalue
// CHECK:     |       | | |   `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int'
    bar2.p = bar.p;
// CHECK:     `-BinaryOperator {{.+}} 'int' '='
// CHECK:       |-MemberExpr {{.+}} .size
// CHECK:       | `-DeclRefExpr {{.+}} [[var_bar2]]
// CHECK:       `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
    bar2.size = 10;
}

// CHECK: |-FunctionDecl [[func_unsizedSizedByToSingle:0x[^ ]+]] {{.+}} unsizedSizedByToSingle
// CHECK: | |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
void unsizedSizedByToSingle(struct unsized * __sized_by(len) p, int len) {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_1:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'struct unsized *__single' <BoundsSafetyPointerCast>
// CHECK: |         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK: |           | | |-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |           | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK: |           | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |           | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |           | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK: |           | | |   |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |           | | |   `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK: |           | |-OpaqueValueExpr [[ove_16]]
// CHECK: |           | | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single' <LValueToRValue>
// CHECK: |           | |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK: |           | `-OpaqueValueExpr [[ove_17]]
// CHECK: |           |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |           |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: |           |-OpaqueValueExpr [[ove_16]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |           `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
    struct unsized * __single p2 = p;
}
// CHECK: |-RecordDecl
struct other;
// CHECK: |-FunctionDecl [[func_unsizedSingleToSingleTypecast:0x[^ ]+]] {{.+}} unsizedSingleToSingleTypecast
// CHECK:   |-ParmVarDecl [[var_p_3:0x[^ ]+]]
void unsizedSingleToSingleTypecast(struct unsized * p) {
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_2:0x[^ ]+]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'struct other *__single' <BitCast>
// CHECK:           `-ImplicitCastExpr {{.+}} 'struct unsized *__single' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_p_3]]
    struct other * __single p2 = p; // expected-warning{{incompatible pointer types initializing 'struct other *__single' with an expression of type 'struct unsized *__single'}}
}

// CHECK: |-FunctionDecl [[func_unsizedSizedByToSizedByTypecast:0x[^ ]+]] {{.+}} unsizedSizedByToSizedByTypecast
// CHECK:   |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
void unsizedSizedByToSizedByTypecast(struct unsized * __sized_by(len) p, int len) {
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_size_1:0x[^ ]+]]
// CHECK:     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   | `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK:     |   `-DependerDeclsAttr
    int size = len;
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_3:0x[^ ]+]]
// CHECK:         `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && size <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__bidi_indexable)p && 0 <= size'
// CHECK:           |-ImplicitCastExpr {{.+}} 'struct other *__single __sized_by(size)':'struct other *__single' <BoundsSafetyPointerCast>
// CHECK:           | `-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           |       | | |-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |       | | |   `-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int'
// CHECK:           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'struct other *' <BoundsSafetyPointerCast>
// CHECK:           | | | | `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           | | | `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | |   |-GetBoundExpr {{.+}} lower
// CHECK:           | |   | `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'struct other *' <BoundsSafetyPointerCast>
// CHECK:           | |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |   | |-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'long'
// CHECK:           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:           |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:           |   |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           |   |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'struct other *__bidi_indexable'
// CHECK:           |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:           |     | `-IntegerLiteral {{.+}} 0
// CHECK:           |     `-OpaqueValueExpr [[ove_21]] {{.*}} 'long'
// CHECK:           |-OpaqueValueExpr [[ove_18]]
// CHECK:           | `-ImplicitCastExpr {{.+}} 'struct other *__bidi_indexable' <BitCast>
// CHECK:           |   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           |     | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK:           |     | | |-OpaqueValueExpr [[ove_19]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |     | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK:           |     | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           |     | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |     | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:           |     | | |   |   `-OpaqueValueExpr [[ove_19]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |     | | |   `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:           |     | |-OpaqueValueExpr [[ove_19]]
// CHECK:           |     | | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single' <LValueToRValue>
// CHECK:           |     | |   `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK:           |     | `-OpaqueValueExpr [[ove_20]]
// CHECK:           |     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           |     |     `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK:           |     |-OpaqueValueExpr [[ove_19]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |     `-OpaqueValueExpr [[ove_20]] {{.*}} 'int'
// CHECK:           `-OpaqueValueExpr [[ove_21]]
// CHECK:             `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                 `-DeclRefExpr {{.+}} [[var_size_1]]
    struct other * __single __sized_by(size) p2 = p; // expected-warning{{incompatible pointer types initializing 'struct other *__single' with an expression of type 'struct unsized *__single __sized_by(len)' (aka 'struct unsized *__single')}} (This warning is redundant and missing __sized_by annotation rdar://112409995)
    // expected-warning@-1{{incompatible pointer types initializing 'struct other *__single __sized_by(size)' (aka 'struct other *__single') with an expression of type 'struct unsized *__single __sized_by(len)' (aka 'struct unsized *__single')}}
}

// CHECK: |-FunctionDecl [[func_voidSizedByToVoidSizedBy:0x[^ ]+]] {{.+}} voidSizedByToVoidSizedBy
// CHECK:   |-ParmVarDecl [[var_p_5:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
void voidSizedByToVoidSizedBy(void * __sized_by(len) p, int len) {
    int size = len;
    void * __single __sized_by(size) p2 = p;
}

// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_size_2:0x[^ ]+]]
// CHECK:     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   | `-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK:     |   `-DependerDeclsAttr
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_4:0x[^ ]+]]
// CHECK:         `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && size <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__bidi_indexable)p && 0 <= size'
// CHECK:           |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(size)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:           | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:           |     | | |-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:           |     | | |   `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int'
// CHECK:           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:           | | | | `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           | | | `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | |   |-GetBoundExpr {{.+}} lower
// CHECK:           | |   | `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:           | |     `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |   | |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'long'
// CHECK:           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:           |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:           |   |   |   `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           |   |       `-OpaqueValueExpr [[ove_22]] {{.*}} 'void *__bidi_indexable'
// CHECK:           |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:           |     | `-IntegerLiteral {{.+}} 0
// CHECK:           |     `-OpaqueValueExpr [[ove_25]] {{.*}} 'long'
// CHECK:           |-OpaqueValueExpr [[ove_22]]
// CHECK:           | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:           |   | | |-OpaqueValueExpr [[ove_23]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:           |   | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:           |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |   | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:           |   | | |   |   `-OpaqueValueExpr [[ove_23]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:           |   | | |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:           |   | |-OpaqueValueExpr [[ove_23]]
// CHECK:           |   | | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(len)':'void *__single' <LValueToRValue>
// CHECK:           |   | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK:           |   | `-OpaqueValueExpr [[ove_24]]
// CHECK:           |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           |   |     `-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK:           |   |-OpaqueValueExpr [[ove_23]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK:           |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int'
// CHECK:           `-OpaqueValueExpr [[ove_25]]
// CHECK:             `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                 `-DeclRefExpr {{.+}} [[var_size_2]]

// CHECK: |-FunctionDecl [[func_unsizedBidiForgedNull:0x[^ ]+]] {{.+}} unsizedBidiForgedNull
void unsizedBidiForgedNull() {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_5:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-CStyleCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BitCast>
// CHECK: |           `-ForgePtrExpr
// CHECK: |             |-ParenExpr
// CHECK: |             | `-IntegerLiteral {{.+}} 0
// CHECK: |             |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |             | `-ParenExpr
// CHECK: |             |   `-IntegerLiteral {{.+}} 10
    struct unsized * __bidi_indexable p2 = __unsafe_forge_bidi_indexable(struct unsized *, 0, 10);
}

// CHECK: |-FunctionDecl [[func_unsizedBidiForgedDyn:0x[^ ]+]] {{.+}} unsizedBidiForgedDyn
// CHECK: | |-ParmVarDecl [[var_p_6:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_5:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
void unsizedBidiForgedDyn(struct unsized * __sized_by(len) p, int len) {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_6:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-CStyleCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BitCast>
// CHECK: |           `-ForgePtrExpr
// CHECK: |             |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |             | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |             | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK: |             | | | |-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |             | | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK: |             | | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |             | | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |             | | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK: |             | | | |   |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |             | | | |   `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'int'
// CHECK: |             | | |-OpaqueValueExpr [[ove_26]]
// CHECK: |             | | | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single' <LValueToRValue>
// CHECK: |             | | |   `-ParenExpr
// CHECK: |             | | |     `-DeclRefExpr {{.+}} [[var_p_6]]
// CHECK: |             | | `-OpaqueValueExpr [[ove_27]]
// CHECK: |             | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |             | |     `-DeclRefExpr {{.+}} [[var_len_5]]
// CHECK: |             | |-OpaqueValueExpr [[ove_26]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK: |             | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: |             |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |             | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |             |   `-ParenExpr
// CHECK: |             |     `-DeclRefExpr {{.+}} [[var_len_5]]
    struct unsized * __bidi_indexable p2 = __unsafe_forge_bidi_indexable(struct unsized *, p, len);
}

// CHECK: |-FunctionDecl [[func_unsizedBidiForgedTypecast:0x[^ ]+]] {{.+}} unsizedBidiForgedTypecast
void unsizedBidiForgedTypecast() {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_7:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'struct other *__bidi_indexable' <BitCast>
// CHECK: |         `-ParenExpr
// CHECK: |           `-CStyleCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BitCast>
// CHECK: |             `-ForgePtrExpr
// CHECK: |               |-ParenExpr
// CHECK: |               | `-IntegerLiteral {{.+}} 0
// CHECK: |               |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |               | `-ParenExpr
// CHECK: |               |   `-IntegerLiteral {{.+}} 10
    struct other * __bidi_indexable p2 = __unsafe_forge_bidi_indexable(struct unsized *, 0, 10); // expected-warning{{incompatible pointer types initializing 'struct other *__bidi_indexable' with an expression of type 'struct unsized *__bidi_indexable'}}
}

// CHECK: |-FunctionDecl [[func_unsizedBidiForgedTypecastToInt:0x[^ ]+]] {{.+}} unsizedBidiForgedTypecastToInt
void unsizedBidiForgedTypecastToInt() {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_8:0x[^ ]+]]
// CHECK: |       `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BitCast>
// CHECK: |         `-ParenExpr
// CHECK: |           `-CStyleCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BitCast>
// CHECK: |             `-ForgePtrExpr
// CHECK: |               |-ParenExpr
// CHECK: |               | `-IntegerLiteral {{.+}} 0
// CHECK: |               |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK: |               | `-ParenExpr
// CHECK: |               |   `-IntegerLiteral {{.+}} 10
    int * __bidi_indexable p2 = __unsafe_forge_bidi_indexable(struct unsized *, 0, 10); // expected-warning{{incompatible pointer types initializing 'int *__bidi_indexable' with an expression of type 'struct unsized *__bidi_indexable'}}
}

// CHECK: |-FunctionDecl [[func_unsizedBidiForgedToSizedBy:0x[^ ]+]] {{.+}} unsizedBidiForgedToSizedBy
// CHECK:   |-ParmVarDecl [[var_p_7:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len_6:0x[^ ]+]]
// CHECK:   | `-DependerDeclsAttr
void unsizedBidiForgedToSizedBy(struct unsized * __sized_by(len) p, int len) {
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_size_3:0x[^ ]+]]
// CHECK:     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   | `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK:     |   `-DependerDeclsAttr
    int size = len;
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_9:0x[^ ]+]]
// CHECK:         `-BoundsCheckExpr {{.+}} '((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len))) <= __builtin_get_pointer_upper_bound(((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len)))) && __builtin_get_pointer_lower_bound(((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len)))) <= ((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len))) && size <= (char *)__builtin_get_pointer_upper_bound(((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len)))) - (char *__bidi_indexable)((struct unsized *__bidi_indexable)__builtin_unsafe_forge_bidi_indexable((p), (len))) && 0 <= size'
// CHECK:           |-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(size)':'struct unsized *__single' <BoundsSafetyPointerCast>
// CHECK:           | `-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           |         | | | |-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |         | | | |   `-OpaqueValueExpr [[ove_30:0x[^ ]+]] {{.*}} 'int'
// CHECK:           |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:           | | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           | | | `-GetBoundExpr {{.+}} upper
// CHECK:           | | |   `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           | |   |-GetBoundExpr {{.+}} lower
// CHECK:           | |   | `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:           | |     `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:           |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |   | |-OpaqueValueExpr [[ove_31:0x[^ ]+]] {{.*}} 'long'
// CHECK:           |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:           |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:           |   |   |   `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:           |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:           |   |       `-OpaqueValueExpr [[ove_28]] {{.*}} 'struct unsized *__bidi_indexable'
// CHECK:           |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:           |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:           |     | `-IntegerLiteral {{.+}} 0
// CHECK:           |     `-OpaqueValueExpr [[ove_31]] {{.*}} 'long'
// CHECK:           |-OpaqueValueExpr [[ove_28]]
// CHECK:           | `-ParenExpr
// CHECK:           |   `-CStyleCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BitCast>
// CHECK:           |     `-ForgePtrExpr
// CHECK:           |       |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |       | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           |       | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct unsized *__bidi_indexable'
// CHECK:           |       | | | |-OpaqueValueExpr [[ove_29]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |       | | | |-ImplicitCastExpr {{.+}} 'struct unsized *' <BitCast>
// CHECK:           |       | | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           |       | | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           |       | | | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *' <BoundsSafetyPointerCast>
// CHECK:           |       | | | |   |   `-OpaqueValueExpr [[ove_29]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |       | | | |   `-OpaqueValueExpr [[ove_30]] {{.*}} 'int'
// CHECK:           |       | | |-OpaqueValueExpr [[ove_29]]
// CHECK:           |       | | | `-ImplicitCastExpr {{.+}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single' <LValueToRValue>
// CHECK:           |       | | |   `-ParenExpr
// CHECK:           |       | | |     `-DeclRefExpr {{.+}} [[var_p_7]]
// CHECK:           |       | | `-OpaqueValueExpr [[ove_30]]
// CHECK:           |       | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           |       | |     `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK:           |       | |-OpaqueValueExpr [[ove_29]] {{.*}} 'struct unsized *__single __sized_by(len)':'struct unsized *__single'
// CHECK:           |       | `-OpaqueValueExpr [[ove_30]] {{.*}} 'int'
// CHECK:           |       |-ImplicitCastExpr {{.+}} 'unsigned long' <IntegralCast>
// CHECK:           |       | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           |       |   `-ParenExpr
// CHECK:           |       |     `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK:           `-OpaqueValueExpr [[ove_31]]
// CHECK:             `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:               `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                 `-DeclRefExpr {{.+}} [[var_size_3]]
    struct unsized * __single __sized_by(size) p2 = __unsafe_forge_bidi_indexable(struct unsized *, p, len);
}

// CHECK: |-FunctionDecl [[func_unsizedSingleForgedNull:0x[^ ]+]] {{.+}} unsizedSingleForgedNull
void unsizedSingleForgedNull() {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_10:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-CStyleCastExpr {{.+}} 'struct unsized *__single' <BitCast>
// CHECK: |           `-ForgePtrExpr
// CHECK: |             |-ParenExpr
// CHECK: |             | `-IntegerLiteral {{.+}} 0
    struct unsized * __single p2 = __unsafe_forge_single(struct unsized *, 0);
}

// CHECK: |-FunctionDecl [[func_unsizedSingleForgedDyn:0x[^ ]+]] {{.+}} unsizedSingleForgedDyn
// CHECK: | |-ParmVarDecl [[var_p_8:0x[^ ]+]]
void unsizedSingleForgedDyn(int p) {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_11:0x[^ ]+]]
// CHECK: |       `-ParenExpr
// CHECK: |         `-CStyleCastExpr {{.+}} 'struct unsized *__single' <BitCast>
// CHECK: |           `-ForgePtrExpr
// CHECK: |             |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |             | `-ParenExpr
// CHECK: |             |   `-DeclRefExpr {{.+}} [[var_p_8]]
    struct unsized * __single p2 = __unsafe_forge_single(struct unsized *, p);
}

// CHECK: |-FunctionDecl [[func_unsizedSingleForgedToBidi:0x[^ ]+]] {{.+}} unsizedSingleForgedToBidi
// CHECK:   |-ParmVarDecl [[var_p_9:0x[^ ]+]]
void unsizedSingleForgedToBidi(int p) {
// CHECK:   `-CompoundStmt
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p2_12:0x[^ ]+]]
// CHECK:         `-RecoveryExpr
// CHECK:           `-ParenExpr
// CHECK:             `-CStyleCastExpr {{.+}} 'struct unsized *__single' <BitCast>
// CHECK:               `-ForgePtrExpr
// CHECK:                 |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                 | `-ParenExpr
// CHECK:                 |   `-DeclRefExpr {{.+}} [[var_p_9]]
    // expected-note@+1{{pointer 'p2' declared here}}
    struct unsized * __bidi_indexable p2 = __unsafe_forge_single(struct unsized *, p); // expected-error{{cannot initialize indexable pointer with type 'struct unsized *__bidi_indexable' from __single pointer to incomplete type 'struct unsized *__single'; consider declaring pointer 'p2' as '__single'}}
}

// CHECK: |-FunctionDecl [[func_unsizedSingleForgedToSizedBy:0x[^ ]+]] {{.+}} unsizedSingleForgedToSizedBy
// CHECK:   |-ParmVarDecl [[var_p_10:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_len_7:0x[^ ]+]]
void unsizedSingleForgedToSizedBy(int p, int len) {
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_size_4:0x[^ ]+]]
// CHECK:     |   |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:     |   | `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK:     |   `-DependerDeclsAttr
    int size = len; // expected-note{{size initialized here}}
// CHECK:    `-DeclStmt
// CHECK:      `-VarDecl [[var_p2_13:0x[^ ]+]]
// CHECK:        `-BoundsCheckExpr {{.+}} '(struct unsized *__single)__builtin_unsafe_forge_single((p)) <= __builtin_get_pointer_upper_bound((struct unsized *__single)__builtin_unsafe_forge_single((p))) && __builtin_get_pointer_lower_bound((struct unsized *__single)__builtin_unsafe_forge_single((p))) <= (struct unsized *__single)__builtin_unsafe_forge_single((p)) && size <= (char *)__builtin_get_pointer_upper_bound((struct unsized *__single)__builtin_unsafe_forge_single((p))) - (char *__single)(struct unsized *__single)__builtin_unsafe_forge_single((p)) && 0 <= size'
// CHECK:          |-ParenExpr
// CHECK:          | `-OpaqueValueExpr [[ove_32:0x[^ ]+]] {{.*}} 'struct unsized *__single'
// CHECK:          |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:          | | | |-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          | | | `-GetBoundExpr {{.+}} upper
// CHECK:          | | |   `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:          | | |     `-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:          | |   |-GetBoundExpr {{.+}} lower
// CHECK:          | |   | `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:          | |   |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          | |   `-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:          |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:          |   | |-OpaqueValueExpr [[ove_33:0x[^ ]+]] {{.*}} 'long'
// CHECK:          |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:          |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:          |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:          |   |   |   `-ImplicitCastExpr {{.+}} 'struct unsized *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:          |   |   |     `-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          |   |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK:          |   |     `-OpaqueValueExpr [[ove_32]] {{.*}} 'struct unsized *__single'
// CHECK:          |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:          |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:          |     | `-IntegerLiteral {{.+}} 0
// CHECK:          |     `-OpaqueValueExpr [[ove_33]] {{.*}} 'long'
// CHECK:          |-OpaqueValueExpr [[ove_32]]
// CHECK:          | `-CStyleCastExpr {{.+}} 'struct unsized *__single' <BitCast>
// CHECK:          |   `-ForgePtrExpr
// CHECK:          |     |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:          |     | `-ParenExpr
// CHECK:          |     |   `-DeclRefExpr {{.+}} [[var_p_10]]
// CHECK:          `-OpaqueValueExpr [[ove_33]]
// CHECK:            `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:              `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:                `-DeclRefExpr {{.+}} [[var_size_4]]
    struct unsized * __single __sized_by(size) p2 = __unsafe_forge_single(struct unsized *, p); // expected-warning{{size value is not statically known: initializing 'p2' of type 'struct unsized *__single __sized_by(size)' (aka 'struct unsized *__single') with 'struct unsized *__single' is invalid for any size greater than 0}}
}

// CHECK: |-FunctionDecl [[func_unsizedSingleForgedToBidiVoid:0x[^ ]+]] {{.+}} unsizedSingleForgedToBidiVoid
// CHECK: | |-ParmVarDecl [[var_p_11:0x[^ ]+]]
void unsizedSingleForgedToBidiVoid(int p) {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_14:0x[^ ]+]]
// CHECK: |       `-RecoveryExpr
// CHECK: |         `-ParenExpr
// CHECK: |           `-CStyleCastExpr {{.+}} 'void *__single' <NoOp>
// CHECK: |             `-ForgePtrExpr
// CHECK: |               |-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |               | `-ParenExpr
// CHECK: |               |   `-DeclRefExpr {{.+}} [[var_p_11]]
    // expected-note@+1{{pointer 'p2' declared here}}
    void * __bidi_indexable p2 = __unsafe_forge_single(void *, p); // expected-error{{cannot initialize indexable pointer with type 'void *__bidi_indexable' from __single pointer to incomplete type 'void *__single'; consider declaring pointer 'p2' as '__single'}}
}

// CHECK: |-FunctionDecl [[func_unsizedSizedByToBidiVoid:0x[^ ]+]] {{.+}} unsizedSizedByToBidiVoid
// CHECK: | |-ParmVarDecl [[var_p_12:0x[^ ]+]]
// CHECK: | |-ParmVarDecl [[var_len_8:0x[^ ]+]]
// CHECK: | | `-DependerDeclsAttr
void unsizedSizedByToBidiVoid(void * __sized_by(len) p, int len) {
// CHECK: | `-CompoundStmt
// CHECK: |   `-DeclStmt
// CHECK: |     `-VarDecl [[var_p2_15:0x[^ ]+]]
// CHECK: |       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: |         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: |         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK: |         | | |-OpaqueValueExpr [[ove_34:0x[^ ]+]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK: |         | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK: |         | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK: |         | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK: |         | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK: |         | | |   |   `-OpaqueValueExpr [[ove_34]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK: |         | | |   `-OpaqueValueExpr [[ove_35:0x[^ ]+]] {{.*}} 'int'
// CHECK: |         | |-OpaqueValueExpr [[ove_34]]
// CHECK: |         | | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(len)':'void *__single' <LValueToRValue>
// CHECK: |         | |   `-DeclRefExpr {{.+}} [[var_p_12]]
// CHECK: |         | `-OpaqueValueExpr [[ove_35]]
// CHECK: |         |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: |         |     `-DeclRefExpr {{.+}} [[var_len_8]]
// CHECK: |         |-OpaqueValueExpr [[ove_34]] {{.*}} 'void *__single __sized_by(len)':'void *__single'
// CHECK: |         `-OpaqueValueExpr [[ove_35]] {{.*}} 'int'
    void * __bidi_indexable p2 = p;
}

// CHECK:  -FunctionDecl [[func_unsizedSingleToSizedByToBidiVoid:0x[^ ]+]] {{.+}} unsizedSingleToSizedByToBidiVoid
// CHECK:   |-ParmVarDecl [[var_p_13:0x[^ ]+]]
void unsizedSingleToSizedByToBidiVoid(void * p) { // rdar://112462891
// CHECK:   `-CompoundStmt
// CHECK:     |-DeclStmt
// CHECK:     | `-VarDecl [[var_p2_16:0x[^ ]+]]
// CHECK:     |   `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && 0 <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__single)p && 0 <= 0'
// CHECK:     |     |-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK:     |     | `-DeclRefExpr {{.+}} [[var_p_13]]
// CHECK:     |     |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | | | |-OpaqueValueExpr [[ove_36:0x[^ ]+]] {{.*}} 'void *__single'
// CHECK:     |     | | | `-GetBoundExpr {{.+}} upper
// CHECK:     |     | | |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     | | |     `-OpaqueValueExpr [[ove_36]] {{.*}} 'void *__single'
// CHECK:     |     | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     | |   |-GetBoundExpr {{.+}} lower
// CHECK:     |     | |   | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     | |   |   `-OpaqueValueExpr [[ove_36]] {{.*}} 'void *__single'
// CHECK:     |     | |   `-OpaqueValueExpr [[ove_36]] {{.*}} 'void *__single'
// CHECK:     |     | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:     |     |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |   | |-OpaqueValueExpr [[ove_37:0x[^ ]+]] {{.*}} 'long'
// CHECK:     |     |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:     |     |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:     |     |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:     |     |   |   |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:     |     |   |   |     `-OpaqueValueExpr [[ove_36]] {{.*}} 'void *__single'
// CHECK:     |     |   |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK:     |     |   |     `-OpaqueValueExpr [[ove_36]] {{.*}} 'void *__single'
// CHECK:     |     |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:     |     |     |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |     |     | `-IntegerLiteral {{.+}} 0
// CHECK:     |     |     `-OpaqueValueExpr [[ove_37]] {{.*}} 'long'
// CHECK:     |     |-OpaqueValueExpr [[ove_36]]
// CHECK:     |     | `-ImplicitCastExpr {{.+}} 'void *__single' <LValueToRValue>
// CHECK:     |     |   `-DeclRefExpr {{.+}} [[var_p_13]]
// CHECK:     |     `-OpaqueValueExpr [[ove_37]]
// CHECK:     |       `-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:     |         `-IntegerLiteral {{.+}} 0
    void * __single __sized_by(0) p2 = p;
// CHECK:     `-DeclStmt
// CHECK:       `-VarDecl [[var_p3:0x[^ ]+]]
// CHECK:         `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           | |-BoundsSafetyPointerPromotionExpr {{.+}} 'void *__bidi_indexable'
// CHECK:           | | |-OpaqueValueExpr [[ove_38:0x[^ ]+]] {{.*}} 'void *__single __sized_by(0)':'void *__single'
// CHECK:           | | |-ImplicitCastExpr {{.+}} 'void *' <BitCast>
// CHECK:           | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:           | | |   | `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:           | | |   |   `-OpaqueValueExpr [[ove_38]] {{.*}} 'void *__single __sized_by(0)':'void *__single'
// CHECK:           | | |   `-OpaqueValueExpr [[ove_39:0x[^ ]+]] {{.*}} 'int'
// CHECK:           | |-OpaqueValueExpr [[ove_38]]
// CHECK:           | | `-ImplicitCastExpr {{.+}} 'void *__single __sized_by(0)':'void *__single' <LValueToRValue>
// CHECK:           | |   `-DeclRefExpr {{.+}} [[var_p2_16]]
// CHECK:           | `-OpaqueValueExpr [[ove_39]]
// CHECK:           |   `-IntegerLiteral {{.+}} 0
// CHECK:           |-OpaqueValueExpr [[ove_38]] {{.*}} 'void *__single __sized_by(0)':'void *__single'
// CHECK:           `-OpaqueValueExpr [[ove_39]] {{.*}} 'int'
    void * __bidi_indexable p3 = p2;
}

