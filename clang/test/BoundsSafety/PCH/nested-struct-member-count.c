// Test without pch.
// RUN: %clang_cc1 -fbounds-safety -include %s %s -ast-dump 2>&1 | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -fbounds-safety -emit-pch -o %t %s
// RUN: %clang_cc1 -fbounds-safety -include-pch %t %s -ast-dump 2>&1 | FileCheck %s

#ifndef HEADER
#define HEADER
#include <ptrcheck.h>

struct Inner {
    int dummy;
    int len;
};

struct Outer {
    struct Inner hdr;
    char fam[__counted_by(hdr.len)];
};

#else

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

#endif
