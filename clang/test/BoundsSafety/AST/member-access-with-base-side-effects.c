

// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct foo {
    int *__counted_by(count) p;
    int count;
};

struct foo *bar(void);
// CHECK: |-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar

void baz(void) {
    (void)bar()->p;
}

// CHECK-LABEL: baz
// CHECK:   `-CompoundStmt
// CHECK:     `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK:       `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK:         |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK:         | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK:         | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | |     `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'struct foo *__single'
// CHECK:         | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK:         | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:         | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
// CHECK:         | | | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int'
// CHECK:         | |-OpaqueValueExpr [[ove_1]]
// CHECK:         | | `-CallExpr
// CHECK:         | |   `-ImplicitCastExpr {{.+}} 'struct foo *__single(*__single)(void)' <FunctionToPointerDecay>
// CHECK:         | |     `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK:         | |-OpaqueValueExpr [[ove_2]]
// CHECK:         | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:         | |   `-MemberExpr {{.+}} ->count
// CHECK:         | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct foo *__single'
// CHECK:         | `-OpaqueValueExpr [[ove]]
// CHECK:         |   `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <LValueToRValue>
// CHECK:         |     `-MemberExpr {{.+}} ->p
// CHECK:         |       `-OpaqueValueExpr [[ove_1]] {{.*}} 'struct foo *__single'
// CHECK:         |-OpaqueValueExpr [[ove_1]] {{.*}} 'struct foo *__single'
// CHECK:         |-OpaqueValueExpr [[ove_2]] {{.*}} 'int'
// CHECK:         `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by(count)':'int *__single'
