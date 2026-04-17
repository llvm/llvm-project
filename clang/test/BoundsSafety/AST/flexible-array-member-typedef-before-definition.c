// RUN: %clang_cc1 -fbounds-safety -ast-dump -Wno-bounds-attributes-implicit-conversion-single-to-explicit-indexable %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -ast-dump -Wno-bounds-attributes-implicit-conversion-single-to-explicit-indexable %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

struct S;
typedef struct S *S_ptr;

struct S {
    int count;
    int fam[__counted_by(count)];
};

int foo(struct S *__bidi_indexable p) { return (*p).count; }
// CHECK:      {{^}}|-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       `-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|         `-ParenExpr
// CHECK-NEXT: {{^}}|           `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|             `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|               |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|               | |-PredefinedBoundsCheckExpr {{.+}} 'struct S *__bidi_indexable' <FlexibleArrayCountDeref(BasePtr, FamPtr, Count)>
// CHECK-NEXT: {{^}}|               | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | |-OpaqueValueExpr [[ove]] {{.*}} 'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|               | | | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}|               | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|               | |   `-MemberExpr {{.+}} ->count
// CHECK-NEXT: {{^}}|               | |     `-OpaqueValueExpr [[ove]] {{.*}} 'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | `-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|               |   `-ImplicitCastExpr {{.+}} 'struct S *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|               |     `-DeclRefExpr {{.+}} [[var_p]]
// CHECK-NEXT: {{^}}|               `-OpaqueValueExpr [[ove]] {{.*}} 'struct S *__bidi_indexable'

int bar(S_ptr __bidi_indexable p) { return (*p).count; }
// CHECK:      {{^}}|-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-ReturnStmt
// CHECK-NEXT: {{^}}|     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|       `-MemberExpr {{.+}} .count
// CHECK-NEXT: {{^}}|         `-ParenExpr
// CHECK-NEXT: {{^}}|           `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|             `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|               |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|               | |-PredefinedBoundsCheckExpr {{.+}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable' <FlexibleArrayCountDeref(BasePtr, FamPtr, Count)>
// CHECK-NEXT: {{^}}|               | | |-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | |-OpaqueValueExpr [[ove_1]] {{.*}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | |-ImplicitCastExpr {{.+}} 'int *' <ArrayToPointerDecay>
// CHECK-NEXT: {{^}}|               | | | `-MemberExpr {{.+}} ->fam
// CHECK-NEXT: {{^}}|               | | |   `-OpaqueValueExpr [[ove_1]] {{.*}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|               | |   `-MemberExpr {{.+}} ->count
// CHECK-NEXT: {{^}}|               | |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable'
// CHECK:      {{^}}|               | `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|               |   `-ImplicitCastExpr {{.+}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|               |     `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}|               `-OpaqueValueExpr [[ove_1]] {{.*}} 'S_ptr __bidi_indexable':'struct S *__bidi_indexable'

int baz(S_ptr p) { return foo(p); }
// CHECK:      {{^}}`-FunctionDecl [[func_baz:0x[^ ]+]] {{.+}} baz
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-CallExpr
// CHECK-NEXT: {{^}}        |-ImplicitCastExpr {{.+}} 'int (*__single)(struct S *__bidi_indexable)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}        | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}        `-ImplicitCastExpr {{.+}} 'struct S *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}          `-ImplicitCastExpr {{.+}} 'struct S *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}            `-DeclRefExpr {{.+}} [[var_p_2]]

