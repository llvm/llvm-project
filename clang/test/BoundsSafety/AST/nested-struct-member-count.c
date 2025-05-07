
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct Inner {
    int dummy;
    int len;
};

struct Outer {
    struct Inner hdr;
    char fam[__counted_by(hdr.len)];
};

char access(struct Outer *bar, int index) {
    return bar->fam[index];
}

// CHECK:  -FunctionDecl [[func_access:0x[^ ]+]] {{.+}} access
// CHECK:   |-ParmVarDecl [[var_bar:0x[^ ]+]]
// CHECK:   |-ParmVarDecl [[var_index:0x[^ ]+]]
// CHECK:   `-CompoundStmt
// CHECK:     `-ReturnStmt
// CHECK:       `-ImplicitCastExpr {{.+}} 'char' <LValueToRValue>
// CHECK:         `-ArraySubscriptExpr
// CHECK:           |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:           | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:           | | |-BoundsSafetyPointerPromotionExpr {{.+}} 'char *__bidi_indexable'
// CHECK:           | | | |-ImplicitCastExpr {{.+}} 'char *' <ArrayToPointerDecay>
// CHECK:           | | | | `-MemberExpr {{.+}} ->fam
// CHECK:           | | | |   `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct Outer *__single'
// CHECK:           | | | |-GetBoundExpr {{.+}} upper
// CHECK:           | | | | `-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK:           | | | |   |-OpaqueValueExpr [[ove]] {{.*}} 'struct Outer *__single'
// CHECK:           | | | |   |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:           | | | |   | |-ImplicitCastExpr {{.+}} 'char *' <ArrayToPointerDecay>
// CHECK:           | | | |   | | `-MemberExpr {{.+}} ->fam
// CHECK:           | | | |   | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct Outer *__single'
// CHECK:           | | | |   | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           | | | |   |   `-MemberExpr {{.+}} .len
// CHECK:           | | | |   |     `-MemberExpr {{.+}} ->hdr
// CHECK:           | | | |   |       `-OpaqueValueExpr [[ove]] {{.*}} 'struct Outer *__single'
// CHECK:           | | `-OpaqueValueExpr [[ove]]
// CHECK:           | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK:           | |     `-DeclRefExpr {{.+}} [[var_bar]]
// CHECK:           | `-OpaqueValueExpr [[ove]] {{.*}} 'struct Outer *__single'
// CHECK:           `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_index]]

struct Outer * assign(void * __bidi_indexable bar, int len) {
    struct Outer * __single s = (struct Outer *) bar;
    s->hdr.len = len;
    return s;
}

// CHECK: -FunctionDecl [[func_assign:0x[^ ]+]] {{.+}} assign
// CHECK:  |-ParmVarDecl [[var_bar_1:0x[^ ]+]]
// CHECK:  |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK:  `-CompoundStmt
// CHECK:    |-DeclStmt
// CHECK:    | `-VarDecl [[var_s:0x[^ ]+]]
// CHECK:    |   `-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:    |     |-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK:    |     | `-PredefinedBoundsCheckExpr {{.+}} 'struct Outer *__bidi_indexable' <FlexibleArrayCountAssign(BasePtr, FamPtr, Count)>
// CHECK:    |     |   |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:    |     |   |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:    |     |   |-ImplicitCastExpr {{.+}} 'char *' <ArrayToPointerDecay>
// CHECK:    |     |   | `-MemberExpr {{.+}} ->fam
// CHECK:    |     |   |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:    |     |   `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:    |     |-OpaqueValueExpr [[ove_1]]
// CHECK:    |     | `-CStyleCastExpr {{.+}} 'struct Outer *__bidi_indexable' <BitCast>
// CHECK:    |     |   `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <LValueToRValue>
// CHECK:    |     |     `-DeclRefExpr {{.+}} [[var_bar_1]]
// CHECK:    |     `-OpaqueValueExpr [[ove_2]]
// CHECK:    |       `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:    |         `-DeclRefExpr {{.+}} [[var_len]]
// CHECK:    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:    | |-BinaryOperator {{.+}} 'int' '='
// CHECK:    | | |-MemberExpr {{.+}} .len
// CHECK:    | | | `-MemberExpr {{.+}} ->hdr
// CHECK:    | | |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK:    | | |     `-DeclRefExpr {{.+}} [[var_s]]
// CHECK:    | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:    | |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct Outer *__bidi_indexable'
// CHECK:    | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:    `-ReturnStmt
// CHECK:      `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <BoundsSafetyPointerCast>
// CHECK:        `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:          |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:          | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct Outer *__bidi_indexable'
// CHECK:          | | |-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'struct Outer *__single'
// CHECK:          | | |-BinaryOperator {{.+}} 'char *' '+'
// CHECK:          | | | |-ImplicitCastExpr {{.+}} 'char *' <ArrayToPointerDecay>
// CHECK:          | | | | `-MemberExpr {{.+}} ->fam
// CHECK:          | | | |   `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single'
// CHECK:          | | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:          | | |   `-MemberExpr {{.+}} .len
// CHECK:          | | |     `-MemberExpr {{.+}} ->hdr
// CHECK:          | | |       `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single'
// CHECK:          | `-OpaqueValueExpr [[ove_3]]
// CHECK:          |   `-ImplicitCastExpr {{.+}} 'struct Outer *__single' <LValueToRValue>
// CHECK:          |     `-DeclRefExpr {{.+}} [[var_s]]
// CHECK:          `-OpaqueValueExpr [[ove_3]] {{.*}} 'struct Outer *__single'

