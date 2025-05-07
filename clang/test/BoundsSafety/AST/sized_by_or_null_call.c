
// RUN: %clang_cc1 -ast-dump -fbounds-safety -Wno-bounds-safety-init-list %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-bounds-safety-init-list %s | FileCheck %s

#include <ptrcheck.h>

// CHECK:      {{^}}|-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| `-CompoundStmt
void foo(int *__sized_by_or_null(len) p, int len) {}

// CHECK: {{^}}|-FunctionDecl [[func_caller_1:0x[^ ]+]] {{.+}} caller_1
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '0 <= 0 && 0 <= 0 && !0 || 2 <= (char *)0 - (char *)0 && 0 <= 2'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|     |   `-IntegerLiteral {{.+}} 2
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
void caller_1() {
  foo(0, 2);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_2:0x[^ ]+]] {{.+}} caller_2
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '0 <= 0 && 0 <= 0 && !0 || 0 <= (char *)0 - (char *)0 && 0 <= 0'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|     |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
void caller_2() {
  foo(0, 0);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_3:0x[^ ]+]] {{.+}} caller_3
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= (char *)__builtin_get_pointer_upper_bound(p) - (char *)p && 0 <= len'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | | | |     | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | | | |     | | |   `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |       `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_4]]
// CHECK-NEXT: {{^}}|     | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|     | |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|     | |   | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | |   |   `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|     | |   | |-OpaqueValueExpr [[ove_5]]
// CHECK-NEXT: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   | |   `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK-NEXT: {{^}}|     | |   | `-OpaqueValueExpr [[ove_6]]
// CHECK-NEXT: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK-NEXT: {{^}}|     | |   |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __sized_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_7]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
void caller_3(int *__sized_by_or_null(len) p, int len) {
  foo(p, len);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_4:0x[^ ]+]] {{.+}} caller_4
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_i:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '&i <= __builtin_get_pointer_upper_bound(&i) && __builtin_get_pointer_lower_bound(&i) <= &i && !&i || -1 <= (char *)__builtin_get_pointer_upper_bound(&i) - (char *)&i && 0 <= -1'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_8]]
// CHECK-NEXT: {{^}}|     | | `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_i]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_9]]
// CHECK-NEXT: {{^}}|     |   `-UnaryOperator {{.+}} prefix '-'
// CHECK-NEXT: {{^}}|     |     `-IntegerLiteral {{.+}} 1
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
void caller_4() {
    int i = 0;
    foo(&i, -1);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_5:0x[^ ]+]] {{.+}} caller_5
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   |-DeclStmt
// CHECK-NEXT: {{^}}|   | `-VarDecl [[var_i_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '&i <= __builtin_get_pointer_upper_bound(&i) && __builtin_get_pointer_lower_bound(&i) <= &i && !&i || 2 <= (char *)__builtin_get_pointer_upper_bound(&i) - (char *)&i && 0 <= 2'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |       `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_10]]
// CHECK-NEXT: {{^}}|     | | `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_i_1]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_11]]
// CHECK-NEXT: {{^}}|     |   `-IntegerLiteral {{.+}} 2
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
void caller_5() {
    int i = 0;
    foo(&i, 2);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_6:0x[^ ]+]] {{.+}} caller_6
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= (char *)__builtin_get_pointer_upper_bound(p) - (char *)p && 0 <= len'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | | | |     | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK:      {{^}}|     | | | |     | | |   `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |       `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_12]]
// CHECK-NEXT: {{^}}|     | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}|     | |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}|     | |   | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | |   |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:      {{^}}|     | |   | |-OpaqueValueExpr [[ove_13]]
// CHECK-NEXT: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(len)':'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   | |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK-NEXT: {{^}}|     | |   | `-OpaqueValueExpr [[ove_14]]
// CHECK-NEXT: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|     | |   |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __sized_by(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK:      {{^}}|     | `-OpaqueValueExpr [[ove_15]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
void caller_6(int *__sized_by(len) p, int len) {
  foo(p, len);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_7:0x[^ ]+]] {{.+}} caller_7
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= (char *)__builtin_get_pointer_upper_bound(p) - (char *)p && 0 <= len'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | | | | `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |       `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_16]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_17]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
void caller_7(int *__bidi_indexable p, int len) {
  foo(p, len);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_8:0x[^ ]+]] {{.+}} caller_8
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= (char *)__builtin_get_pointer_upper_bound(p) - (char *)p && 0 <= len'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |   | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   | |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}|     | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |   |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   | `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |   |   `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}|     | |       |   |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}|     | |       |   |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |       |   `-CStyleCastExpr {{.+}} 'char *__single' <BitCast>
// CHECK-NEXT: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_18]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_19]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
void caller_8(int *__single p, int len) {
  foo(p, len);
}

// CHECK:      {{^}}|-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_out:0x[^ ]+]]
// CHECK-NEXT: {{^}}| `-ParmVarDecl [[var_len_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}|   `-DependerDeclsAttr
void bar(int *__sized_by(*len) *out, int *len);

// CHECK-NEXT: {{^}}|-FunctionDecl [[func_caller_9:0x[^ ]+]] {{.+}} caller_9
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_out_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}| |-ParmVarDecl [[var_len_6:0x[^ ]+]]
// CHECK-NEXT: {{^}}| | `-DependerDeclsAttr
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-CallExpr
// CHECK-NEXT: {{^}}|     | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(*len)*__single, int *__single)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK-NEXT: {{^}}|     | | |-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int *__single __sized_by(*len)*__single'
// CHECK:      {{^}}|     | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_20]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by(*len)*__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_out_1]]
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_21]]
// CHECK-NEXT: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_20]] {{.*}} 'int *__single __sized_by(*len)*__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__single'
void caller_9(int *__sized_by(*len) *out, int *len){
    bar(out, len);
}

// CHECK:      {{^}}`-FunctionDecl [[func_caller_10:0x[^ ]+]] {{.+}} caller_10
// CHECK-NEXT: {{^}}  |-ParmVarDecl [[var_len_7:0x[^ ]+]]
// CHECK-NEXT: {{^}}  `-CompoundStmt
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_count:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |   `-DependerDeclsAttr
// CHECK-NEXT: {{^}}    |-DeclStmt
// CHECK-NEXT: {{^}}    | `-VarDecl [[var_p_5:0x[^ ]+]]
// CHECK-NEXT: {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}    | | |-CallExpr
// CHECK-NEXT: {{^}}    | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __sized_by(*len)*__single, int *__single)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}    | | | | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK-NEXT: {{^}}    | | | |-ImplicitCastExpr {{.+}} 'int *__single __sized_by(*len)*__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'int *__single __sized_by_or_null(count)*__bidi_indexable'
// CHECK:      {{^}}    | | | `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}    | | |   `-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    | | |-OpaqueValueExpr [[ove_22]]
// CHECK-NEXT: {{^}}    | | | `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}    | | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK-NEXT: {{^}}    | | `-OpaqueValueExpr [[ove_23]]
// CHECK-NEXT: {{^}}    | |   `-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK-NEXT: {{^}}    | |-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__single __sized_by_or_null(count)*__bidi_indexable'
// CHECK:      {{^}}    | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}    `-ReturnStmt
// CHECK-NEXT: {{^}}      `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__bidi_indexable)p && 0 <= len'
// CHECK-NEXT: {{^}}        |-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |     | | |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'int *__single __sized_by_or_null(count)':'int *__single'
// CHECK:      {{^}}        |     | | |   `-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | | | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        | | |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        | |   |-GetBoundExpr {{.+}} lower
// CHECK-NEXT: {{^}}        | |   | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        | |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        | `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}        |   |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}        |   | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}        |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}        |     | | `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'int'
// CHECK:      {{^}}        |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}        |     |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}        |     |   | `-GetBoundExpr {{.+}} upper
// CHECK-NEXT: {{^}}        |     |   |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |     |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |     |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK-NEXT: {{^}}        |     |       `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK:      {{^}}        |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}        |       |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}        |       `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK:      {{^}}        |-OpaqueValueExpr [[ove_24]]
// CHECK-NEXT: {{^}}        | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}        |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}        |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK-NEXT: {{^}}        |   | | |-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __sized_by_or_null(count)':'int *__single'
// CHECK:      {{^}}        |   | | |-ImplicitCastExpr {{.+}} 'int *' <BitCast>
// CHECK-NEXT: {{^}}        |   | | | `-BinaryOperator {{.+}} 'char *' '+'
// CHECK-NEXT: {{^}}        |   | | |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK-NEXT: {{^}}        |   | | |   | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK-NEXT: {{^}}        |   | | |   |   `-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __sized_by_or_null(count)':'int *__single'
// CHECK:      {{^}}        |   | | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK:      {{^}}        |   | |-OpaqueValueExpr [[ove_25]]
// CHECK-NEXT: {{^}}        |   | | `-ImplicitCastExpr {{.+}} 'int *__single __sized_by_or_null(count)':'int *__single' <LValueToRValue>
// CHECK-NEXT: {{^}}        |   | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK-NEXT: {{^}}        |   | `-OpaqueValueExpr [[ove_26]]
// CHECK-NEXT: {{^}}        |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}        |   |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK-NEXT: {{^}}        |   |-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __sized_by_or_null(count)':'int *__single'
// CHECK:      {{^}}        |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK:      {{^}}        `-OpaqueValueExpr [[ove_27]]
// CHECK-NEXT: {{^}}          `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK-NEXT: {{^}}            `-DeclRefExpr {{.+}} [[var_len_7]]
int *__sized_by_or_null(len) caller_10(int len) {
    int count;
    int *__sized_by_or_null(count) p;
    bar(&p, &count);
    return p;
}
