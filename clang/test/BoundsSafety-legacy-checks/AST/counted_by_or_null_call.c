// RUN: %clang_cc1 -ast-dump -fbounds-safety -Wno-bounds-safety-init-list %s | FileCheck %s
// RUN: %clang_cc1 -ast-dump -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -Wno-bounds-safety-init-list %s | FileCheck %s

#include <ptrcheck.h>

// CHECK: {{^}}|-FunctionDecl [[func_foo:0x[^ ]+]] {{.+}} foo
// CHECK: {{^}}| |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_len:0x[^ ]+]]
// CHECK: {{^}}| | `-DependerDeclsAttr
// CHECK: {{^}}| `-CompoundStmt
void foo(int *__counted_by_or_null(len) p, int len) {}

// CHECK: {{^}}|-FunctionDecl [[func_caller_1:0x[^ ]+]] {{.+}} caller_1
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '0 <= 0 && 0 <= 0 && !0 || 2 <= 0 - 0 && 0 <= 2'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-OpaqueValueExpr [[ove:0x[^ ]+]]
// CHECK-NEXT: {{^}}|     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]]
// CHECK-NEXT: {{^}}|     | | |   `-IntegerLiteral {{.+}} 2
// CHECK-NEXT: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_1]]
// CHECK-NEXT: {{^}}|     |   `-IntegerLiteral {{.+}} 2
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
void caller_1() {
  foo(0, 2);
}

// CHECK:      {{^}}|-FunctionDecl [[func_caller_2:0x[^ ]+]] {{.+}} caller_2
// CHECK-NEXT: {{^}}| `-CompoundStmt
// CHECK-NEXT: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK-NEXT: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK-NEXT: {{^}}|     | |-BoundsCheckExpr {{.+}} '0 <= 0 && 0 <= 0 && !0 || 0 <= 0 - 0 && 0 <= 0'
// CHECK-NEXT: {{^}}|     | | |-CallExpr
// CHECK-NEXT: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK-NEXT: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK-NEXT: {{^}}|     | | | |-OpaqueValueExpr [[ove_2:0x[^ ]+]]
// CHECK-NEXT: {{^}}|     | | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]]
// CHECK-NEXT: {{^}}|     | | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |   |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK-NEXT: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK-NEXT: {{^}}|     | |     | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK-NEXT: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK-NEXT: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:      {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK-NEXT: {{^}}|     | |       |   |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK-NEXT: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | |         `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:      {{^}}|     | |-OpaqueValueExpr [[ove_2]]
// CHECK-NEXT: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <NullToPointer>
// CHECK-NEXT: {{^}}|     | |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     | `-OpaqueValueExpr [[ove_3]]
// CHECK-NEXT: {{^}}|     |   `-IntegerLiteral {{.+}} 0
// CHECK-NEXT: {{^}}|     |-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK:      {{^}}|     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
void caller_2() {
  foo(0, 0);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_3:0x[^ ]+]] {{.+}} caller_3
// CHECK: {{^}}|  |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK: {{^}}|  |-ParmVarDecl [[var_len_1:0x[^ ]+]]
// CHECK: {{^}}|  | `-DependerDeclsAttr
// CHECK: {{^}}|  `-CompoundStmt
// CHECK: {{^}}|    `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|      |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|      | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= __builtin_get_pointer_upper_bound(p) - p && 0 <= len'
// CHECK: {{^}}|      | | |-CallExpr
// CHECK: {{^}}|      | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|      | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|      | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | | | | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | | | |     | | |-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}|      | | | |     | | | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|      | | | `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|      | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|      | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|      | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|      | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |   | | | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|      | |   | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|      | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|      | |   |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |   |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|      | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|      | |     | `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|      | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|      | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|      | |       | | `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|      | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|      | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|      | |       |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |       |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|      | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|      | |         `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
// CHECK: {{^}}|      | |-OpaqueValueExpr [[ove_4]]
// CHECK: {{^}}|      | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|      | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|      | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|      | |   | | |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}|      | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|      | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|      | |   | | | | `-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}|      | |   | | | `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|      | |   | |-OpaqueValueExpr [[ove_5]]
// CHECK: {{^}}|      | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <LValueToRValue>
// CHECK: {{^}}|      | |   | |   `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK: {{^}}|      | |   | `-OpaqueValueExpr [[ove_6]]
// CHECK: {{^}}|      | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|      | |   |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: {{^}}|      | |   |-OpaqueValueExpr [[ove_5]] {{.*}} 'int *__single __counted_by_or_null(len)':'int *__single'
// CHECK: {{^}}|      | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int'
// CHECK: {{^}}|      | `-OpaqueValueExpr [[ove_7]]
// CHECK: {{^}}|      |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|      |     `-DeclRefExpr {{.+}} [[var_len_1]]
// CHECK: {{^}}|      |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|      `-OpaqueValueExpr [[ove_7]] {{.*}} 'int'
void caller_3(int *__counted_by_or_null(len) p, int len) {
  foo(p, len);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_4:0x[^ ]+]] {{.+}} caller_4
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_i:0x[^ ]+]]
// CHECK: {{^}}|   |   `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsCheckExpr {{.+}} '&i <= __builtin_get_pointer_upper_bound(&i) && __builtin_get_pointer_lower_bound(&i) <= &i && !&i || -1 <= __builtin_get_pointer_upper_bound(&i) - &i && 0 <= -1'
// CHECK: {{^}}|     | | |-CallExpr
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | | | | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |     | `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |       |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |         `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_8]]
// CHECK: {{^}}|     | | `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_i]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_9]]
// CHECK: {{^}}|     |   `-UnaryOperator {{.+}} prefix '-'
// CHECK: {{^}}|     |     `-IntegerLiteral {{.+}} 1
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_8]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
void caller_4() {
    int i = 0;
    foo(&i, -1);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_5:0x[^ ]+]] {{.+}} caller_5
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   |-DeclStmt
// CHECK: {{^}}|   | `-VarDecl [[var_i_1:0x[^ ]+]]
// CHECK: {{^}}|   |   `-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsCheckExpr {{.+}} '&i <= __builtin_get_pointer_upper_bound(&i) && __builtin_get_pointer_lower_bound(&i) <= &i && !&i || 2 <= __builtin_get_pointer_upper_bound(&i) - &i && 0 <= 2'
// CHECK: {{^}}|     | | |-CallExpr
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | | | | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |     | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |       |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |         `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_10]]
// CHECK: {{^}}|     | | `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_i_1]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_11]]
// CHECK: {{^}}|     |   `-IntegerLiteral {{.+}} 2
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
void caller_5() {
    int i = 0;
    foo(&i, 2);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_6:0x[^ ]+]] {{.+}} caller_6
// CHECK: {{^}}| |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_len_2:0x[^ ]+]]
// CHECK: {{^}}| | `-DependerDeclsAttr
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= __builtin_get_pointer_upper_bound(p) - p && 0 <= len'
// CHECK: {{^}}|     | | |-CallExpr
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | | | | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | | | |     | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|     | | | |     | | | `-OpaqueValueExpr [[ove_14:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | | `-OpaqueValueExpr [[ove_15:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |     | `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |       |   |   `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |         `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_12]]
// CHECK: {{^}}|     | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|     | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}|     | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | | | | `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: {{^}}|     | |   | |-OpaqueValueExpr [[ove_13]]
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(len)':'int *__single' <LValueToRValue>
// CHECK: {{^}}|     | |   | |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK: {{^}}|     | |   | `-OpaqueValueExpr [[ove_14]]
// CHECK: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|     | |   |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: {{^}}|     | |   |-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single __counted_by(len)':'int *__single'
// CHECK: {{^}}|     | |   `-OpaqueValueExpr [[ove_14]] {{.*}} 'int'
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_15]]
// CHECK: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_2]]
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_15]] {{.*}} 'int'
void caller_6(int *__counted_by(len) p, int len) {
  foo(p, len);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_7:0x[^ ]+]] {{.+}} caller_7
// CHECK: {{^}}| |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_len_3:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= __builtin_get_pointer_upper_bound(p) - p && 0 <= len'
// CHECK: {{^}}|     | | |-CallExpr
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | | | | `-OpaqueValueExpr [[ove_16:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | | | `-OpaqueValueExpr [[ove_17:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | | | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |   | |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     | |   |   |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |     | `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK: {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |       |   |   `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |     `-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |         `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_16]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_17]]
// CHECK: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_3]]
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_16]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_17]] {{.*}} 'int'
void caller_7(int *__bidi_indexable p, int len) {
  foo(p, len);
}

// CHECK: {{^}}|-FunctionDecl [[func_caller_8:0x[^ ]+]] {{.+}} caller_8
// CHECK: {{^}}| |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_len_4:0x[^ ]+]]
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= __builtin_get_pointer_upper_bound(p) - p && 0 <= len'
// CHECK: {{^}}|     | | |-CallExpr
// CHECK: {{^}}|     | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by_or_null(len), int)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | | `-DeclRefExpr {{.+}} [[func_foo]]
// CHECK: {{^}}|     | | | |-OpaqueValueExpr [[ove_18:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | | | `-OpaqueValueExpr [[ove_19:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}|     | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   | | |-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |   | | `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |   `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |   | |     `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   | |       `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |   |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   | `-GetBoundExpr {{.+}} lower
// CHECK: {{^}}|     | |   |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |   |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |   |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}|     | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}|     | |     | `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}|     | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}|     | |       | | `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
// CHECK: {{^}}|     | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}|     | |       |   |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}|     | |       |   |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK: {{^}}|     | |       |   |     `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |       |   `-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}|     | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}|     | |         `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_18]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_19]]
// CHECK: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_4]]
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_18]] {{.*}} 'int *__single'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_19]] {{.*}} 'int'
void caller_8(int *__single p, int len) {
  foo(p, len);
}

// CHECK: {{^}}|-FunctionDecl [[func_bar:0x[^ ]+]] {{.+}} bar
// CHECK: {{^}}| |-ParmVarDecl [[var_out:0x[^ ]+]]
// CHECK: {{^}}| `-ParmVarDecl [[var_len_5:0x[^ ]+]]
// CHECK: {{^}}|   `-DependerDeclsAttr
void bar(int *__counted_by(*len) *out, int *len);

// CHECK: {{^}}|-FunctionDecl [[func_caller_9:0x[^ ]+]] {{.+}} caller_9
// CHECK: {{^}}| |-ParmVarDecl [[var_out_1:0x[^ ]+]]
// CHECK: {{^}}| |-ParmVarDecl [[var_len_6:0x[^ ]+]]
// CHECK: {{^}}| | `-DependerDeclsAttr
// CHECK: {{^}}| `-CompoundStmt
// CHECK: {{^}}|   `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}|     |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}|     | |-CallExpr
// CHECK: {{^}}|     | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(*len)*__single, int *__single)' <FunctionToPointerDecay>
// CHECK: {{^}}|     | | | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK: {{^}}|     | | |-OpaqueValueExpr [[ove_20:0x[^ ]+]] {{.*}} 'int *__single __counted_by(*len)*__single'
// CHECK: {{^}}|     | | `-OpaqueValueExpr [[ove_21:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK: {{^}}|     | |-OpaqueValueExpr [[ove_20]]
// CHECK: {{^}}|     | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)*__single' <LValueToRValue>
// CHECK: {{^}}|     | |   `-DeclRefExpr {{.+}} [[var_out_1]]
// CHECK: {{^}}|     | `-OpaqueValueExpr [[ove_21]]
// CHECK: {{^}}|     |   `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK: {{^}}|     |     `-DeclRefExpr {{.+}} [[var_len_6]]
// CHECK: {{^}}|     |-OpaqueValueExpr [[ove_20]] {{.*}} 'int *__single __counted_by(*len)*__single'
// CHECK: {{^}}|     `-OpaqueValueExpr [[ove_21]] {{.*}} 'int *__single'
void caller_9(int *__counted_by(*len) *out, int *len){
    bar(out, len);
}

// CHECK: {{^}}`-FunctionDecl [[func_caller_10:0x[^ ]+]] {{.+}} caller_10
// CHECK: {{^}}  |-ParmVarDecl [[var_len_7:0x[^ ]+]]
// CHECK: {{^}}  `-CompoundStmt
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_count:0x[^ ]+]]
// CHECK: {{^}}    |   `-DependerDeclsAttr
// CHECK: {{^}}    |-DeclStmt
// CHECK: {{^}}    | `-VarDecl [[var_p_5:0x[^ ]+]]
// CHECK: {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    | |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    | | |-CallExpr
// CHECK: {{^}}    | | | |-ImplicitCastExpr {{.+}} 'void (*__single)(int *__single __counted_by(*len)*__single, int *__single)' <FunctionToPointerDecay>
// CHECK: {{^}}    | | | | `-DeclRefExpr {{.+}} [[func_bar]]
// CHECK: {{^}}    | | | |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*len)*__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | | | | `-OpaqueValueExpr [[ove_22:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(count)*__bidi_indexable'
// CHECK: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | | |   `-OpaqueValueExpr [[ove_23:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | | |-OpaqueValueExpr [[ove_22]]
// CHECK: {{^}}    | | | `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}    | | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK: {{^}}    | | `-OpaqueValueExpr [[ove_23]]
// CHECK: {{^}}    | |   `-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}    | |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK: {{^}}    | |-OpaqueValueExpr [[ove_22]] {{.*}} 'int *__single __counted_by_or_null(count)*__bidi_indexable'
// CHECK: {{^}}    | `-OpaqueValueExpr [[ove_23]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    | |-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || len <= __builtin_get_pointer_upper_bound(p) - p && 0 <= len'
// CHECK: {{^}}    | | |-BinaryOperator {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' '='
// CHECK: {{^}}    | | | |-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK: {{^}}    | | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | | |   `-OpaqueValueExpr [[ove_24:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | | |       | | |-OpaqueValueExpr [[ove_25:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}    | | |       | | | `-OpaqueValueExpr [[ove_26:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    | | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |   |-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |   | |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |   | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |   | | | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   | | `-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    | |   | |   `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   | `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |   |   |-GetBoundExpr {{.+}} lower
// CHECK: {{^}}    | |   |   | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |   |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   `-BinaryOperator {{.+}} 'int' '||'
// CHECK: {{^}}    | |     |-UnaryOperator {{.+}} cannot overflow
// CHECK: {{^}}    | |     | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |     `-BinaryOperator {{.+}} 'int' '&&'
// CHECK: {{^}}    | |       |-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |       | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK: {{^}}    | |       | | `-OpaqueValueExpr [[ove_27:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}    | |       | `-BinaryOperator {{.+}} 'long' '-'
// CHECK: {{^}}    | |       |   |-GetBoundExpr {{.+}} upper
// CHECK: {{^}}    | |       |   | `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |       |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |       |     `-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |       `-BinaryOperator {{.+}} 'int' '<='
// CHECK: {{^}}    | |         |-IntegerLiteral {{.+}} 0
// CHECK: {{^}}    | |         `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: {{^}}    | |-OpaqueValueExpr [[ove_24]]
// CHECK: {{^}}    | | `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    | |   |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}    | |   | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}    | |   | | |-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}    | |   | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}    | |   | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}    | |   | | | | `-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}    | |   | | | `-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK: {{^}}    | |   | |-OpaqueValueExpr [[ove_25]]
// CHECK: {{^}}    | |   | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <LValueToRValue>
// CHECK: {{^}}    | |   | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK: {{^}}    | |   | `-OpaqueValueExpr [[ove_26]]
// CHECK: {{^}}    | |   |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    | |   |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK: {{^}}    | |   |-OpaqueValueExpr [[ove_25]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}    | |   `-OpaqueValueExpr [[ove_26]] {{.*}} 'int'
// CHECK: {{^}}    | `-OpaqueValueExpr [[ove_27]]
// CHECK: {{^}}    |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}    |     `-DeclRefExpr {{.+}} [[var_len_7]]
// CHECK: {{^}}    |-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}    | |-BinaryOperator {{.+}} 'int' '='
// CHECK: {{^}}    | | |-DeclRefExpr {{.+}} [[var_count]]
// CHECK: {{^}}    | | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: {{^}}    | |-OpaqueValueExpr [[ove_24]] {{.*}} 'int *__bidi_indexable'
// CHECK: {{^}}    | `-OpaqueValueExpr [[ove_27]] {{.*}} 'int'
// CHECK: {{^}}    `-ReturnStmt
// CHECK: {{^}}      `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(len)':'int *__single' <BoundsSafetyPointerCast>
// CHECK: {{^}}        `-MaterializeSequenceExpr {{.+}} <Unbind>
// CHECK: {{^}}          |-MaterializeSequenceExpr {{.+}} <Bind>
// CHECK: {{^}}          | |-BoundsSafetyPointerPromotionExpr {{.+}} 'int *__bidi_indexable'
// CHECK: {{^}}          | | |-OpaqueValueExpr [[ove_28:0x[^ ]+]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}          | | |-BinaryOperator {{.+}} 'int *' '+'
// CHECK: {{^}}          | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK: {{^}}          | | | | `-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}          | | | `-OpaqueValueExpr [[ove_29:0x[^ ]+]] {{.*}} 'int'
// CHECK: {{^}}          | |-OpaqueValueExpr [[ove_28]]
// CHECK: {{^}}          | | `-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <LValueToRValue>
// CHECK: {{^}}          | |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK: {{^}}          | `-OpaqueValueExpr [[ove_29]]
// CHECK: {{^}}          |   `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK: {{^}}          |     `-DeclRefExpr {{.+}} [[var_count]]
// CHECK: {{^}}          |-OpaqueValueExpr [[ove_28]] {{.*}} 'int *__single __counted_by_or_null(count)':'int *__single'
// CHECK: {{^}}          `-OpaqueValueExpr [[ove_29]] {{.*}} 'int'
int *__counted_by_or_null(len) caller_10(int len) {
    int count;
    int *__counted_by_or_null(count) p;
    bar(&p, &count);
    p = p; // workaround for missing return bounds check
    count = len;
    return p;
}
