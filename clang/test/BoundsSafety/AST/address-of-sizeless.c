
// RUN: %clang_cc1 -ast-dump -fbounds-safety %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct flexible {
    int count;
    int elems[__counted_by(count)];
};

extern struct incomplete incomplete;
// CHECK: {{^}}|-VarDecl [[var_incomplete:0x[^ ]+]]

extern struct flexible flexible;
// CHECK: {{^}}|-VarDecl [[var_flexible:0x[^ ]+]]

extern void void_global;
// CHECK: {{^}}|-VarDecl [[var_void_global:0x[^ ]+]]

extern void function(void);
// CHECK: {{^}}|-FunctionDecl [[func_function:0x[^ ]+]] {{.+}} 'void (void)'

extern int array[];
// CHECK: {{^}}|-VarDecl [[var_array:0x[^ ]+]]

void address_of(void) {
// CHECK: {{^}}`-FunctionDecl [[func_address_of:0x[^ ]+]] {{.+}} address_of

    (void) &incomplete;
// CHECK: {{^}}    |-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: {{^}}    | `-UnaryOperator {{.+}} 'struct incomplete *__single'{{.*}} prefix '&'
// CHECK: {{^}}    |   `-DeclRefExpr {{.+}} [[var_incomplete]]

    (void) &flexible;
// CHECK: {{^}}    |-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: {{^}}    | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'struct flexible *__bidi_indexable'
// CHECK: {{^}}    |   | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}    |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}    |   | | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK: {{^}}    |   | | | | `-MemberExpr {{.+}} ->elems
// CHECK: {{^}}    |   | | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}    |   | | |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |   | | |     `-MemberExpr {{.+}} ->count
// CHECK: {{^}}    |   | | |       `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'
// CHECK: {{^}}    |   | `-OpaqueValueExpr [[ove]]
// CHECK: {{^}}    |   |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}    |   |     `-DeclRefExpr {{.+}} [[var_flexible]]
// CHECK: {{^}}    |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct flexible *__single'

    (void) &void_global;
// CHECK: {{^}}    |-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: {{^}}    | `-UnaryOperator {{.+}} 'void *__single'{{.*}} prefix '&'
// CHECK: {{^}}    |   `-DeclRefExpr {{.+}} [[var_void_global]]

    (void) &function;
// CHECK: {{^}}    |-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: {{^}}    | `-UnaryOperator {{.+}} 'void (*__single)(void)'{{.*}} prefix '&'
// CHECK: {{^}}    |   `-DeclRefExpr {{.+}} [[func_function]]

    (void) &array;
// CHECK: {{^}}    `-CStyleCastExpr {{.+}} 'void' <ToVoid>
// CHECK: {{^}}      `-UnaryOperator {{.+}} 'int (*__single)[]'{{.*}} prefix '&'
// CHECK: {{^}}        `-DeclRefExpr {{.+}} [[var_array]]
}
