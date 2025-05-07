

// RUN: %clang_cc1 -fbounds-safety -fbounds-safety-bringup-missing-checks=return_size -Wno-bounds-safety-single-to-count -ast-dump %s 2>&1 | FileCheck %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -fbounds-safety-bringup-missing-checks=return_size -Wno-bounds-safety-single-to-count -ast-dump %s 2>&1 | FileCheck %s

#include <ptrcheck.h>

// CHECK: FunctionDecl [[func_cb_in_from_bidi:0x[^ ]+]] {{.+}} cb_in_from_bidi
// CHECK: |-ParmVarDecl [[var_count:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && count <= __builtin_get_pointer_upper_bound(p) - p && 0 <= count'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove:0x[^ ]+]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | | | | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-OpaqueValueExpr [[ove_1:0x[^ ]+]] {{.*}} 'int'
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-GetBoundExpr {{.+}} upper
// CHECK:       |   |   | `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       |   |     `-OpaqueValueExpr [[ove]] {{.*}} 'int *__bidi_indexable'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-OpaqueValueExpr [[ove_1]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p]]
// CHECK:       `-OpaqueValueExpr [[ove_1]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_count]]
int *__counted_by(count) cb_in_from_bidi(int count, int *__bidi_indexable p) {
  return p;
}

// CHECK: FunctionDecl [[func_cb_in_from_indexable:0x[^ ]+]] {{.+}} cb_in_from_indexable
// CHECK: |-ParmVarDecl [[var_count_1:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_1:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && count <= __builtin_get_pointer_upper_bound(p) - p && 0 <= count'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_2:0x[^ ]+]] {{.*}} 'int *__indexable'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | | | | `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       | |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       | |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-OpaqueValueExpr [[ove_3:0x[^ ]+]] {{.*}} 'int'
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-GetBoundExpr {{.+}} upper
// CHECK:       |   |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       |   |   |   `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       |   |   `-ImplicitCastExpr {{.+}} 'int *' <BoundsSafetyPointerCast>
// CHECK:       |   |     `-OpaqueValueExpr [[ove_2]] {{.*}} 'int *__indexable'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-OpaqueValueExpr [[ove_3]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove_2]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__indexable' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p_1]]
// CHECK:       `-OpaqueValueExpr [[ove_3]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_count_1]]
int *__counted_by(count) cb_in_from_indexable(int count, int *__indexable p) {
  return p;
}

// CHECK: FunctionDecl [[func_cb_in_from_single:0x[^ ]+]] {{.+}} cb_in_from_single
// CHECK: |-ParmVarDecl [[var_count_2:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_2:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && count <= __builtin_get_pointer_upper_bound(p) - p && 0 <= count'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_4:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | | |     `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       | |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-OpaqueValueExpr [[ove_5:0x[^ ]+]] {{.*}} 'int'
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-GetBoundExpr {{.+}} upper
// CHECK:       |   |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       |   |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       |   |   `-OpaqueValueExpr [[ove_4]] {{.*}} 'int *__single'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-OpaqueValueExpr [[ove_5]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove_4]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p_2]]
// CHECK:       `-OpaqueValueExpr [[ove_5]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_count_2]]
int *__counted_by(count) cb_in_from_single(int count, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_cb_out_from_single:0x[^ ]+]] {{.+}} cb_out_from_single
// CHECK: |-ParmVarDecl [[var_count_3:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_3:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && *count <= __builtin_get_pointer_upper_bound(p) - p && 0 <= *count'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by(*count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_6:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | | |     `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       | |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       |   | |   `-OpaqueValueExpr [[ove_7:0x[^ ]+]] {{.*}} lvalue
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-GetBoundExpr {{.+}} upper
// CHECK:       |   |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       |   |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       |   |   `-OpaqueValueExpr [[ove_6]] {{.*}} 'int *__single'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:       |       `-OpaqueValueExpr [[ove_7]] {{.*}} lvalue
// CHECK:       |-OpaqueValueExpr [[ove_6]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p_3]]
// CHECK:       `-OpaqueValueExpr [[ove_7]]
// CHECK:         `-UnaryOperator {{.+}} cannot overflow
// CHECK:           `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:             `-DeclRefExpr {{.+}} [[var_count_3]]
int *__counted_by(*count) cb_out_from_single(int *__single count, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_sb_from_single:0x[^ ]+]] {{.+}} sb_from_single
// CHECK: |-ParmVarDecl [[var_size:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_4:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && size <= (char *)__builtin_get_pointer_upper_bound(p) - (char *__bidi_indexable)p && 0 <= size'
// CHECK:       |-ImplicitCastExpr {{.+}} 'void *__single __sized_by(size)':'void *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_8:0x[^ ]+]] {{.*}} 'void *__bidi_indexable'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:       | | | | `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       | |   `-ImplicitCastExpr {{.+}} 'void *' <BoundsSafetyPointerCast>
// CHECK:       | |     `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |   |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |   | | `-OpaqueValueExpr [[ove_9:0x[^ ]+]] {{.*}} 'int'
// CHECK:       |   | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |   |   |-CStyleCastExpr {{.+}} 'char *' <BitCast>
// CHECK:       |   |   | `-GetBoundExpr {{.+}} upper
// CHECK:       |   |   |   `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       |   |   `-ImplicitCastExpr {{.+}} 'char *' <BoundsSafetyPointerCast>
// CHECK:       |   |     `-CStyleCastExpr {{.+}} 'char *__bidi_indexable' <BitCast>
// CHECK:       |   |       `-OpaqueValueExpr [[ove_8]] {{.*}} 'void *__bidi_indexable'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     |-IntegerLiteral {{.+}} 0
// CHECK:       |     `-OpaqueValueExpr [[ove_9]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove_8]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'void *__bidi_indexable' <BitCast>
// CHECK:       |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       |     `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       |       `-DeclRefExpr {{.+}} [[var_p_4]]
// CHECK:       `-OpaqueValueExpr [[ove_9]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_size]]
void *__sized_by(size) sb_from_single(int size, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_cbn_in_from_single:0x[^ ]+]] {{.+}} cbn_in_from_single
// CHECK: |-ParmVarDecl [[var_count_4:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_5:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'p <= __builtin_get_pointer_upper_bound(p) && __builtin_get_pointer_lower_bound(p) <= p && !p || count <= __builtin_get_pointer_upper_bound(p) - p && 0 <= count'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __counted_by_or_null(count)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_10:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | | |-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       | | | `-GetBoundExpr {{.+}} upper
// CHECK:       | | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | | |     `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       | | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | |   |-GetBoundExpr {{.+}} lower
// CHECK:       | |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       | |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '||'
// CHECK:       |   |-UnaryOperator {{.+}} cannot overflow
// CHECK:       |   | `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       |   `-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       |     |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |     | |-ImplicitCastExpr {{.+}} 'long' <IntegralCast>
// CHECK:       |     | | `-OpaqueValueExpr [[ove_11:0x[^ ]+]] {{.*}} 'int'
// CHECK:       |     | `-BinaryOperator {{.+}} 'long' '-'
// CHECK:       |     |   |-GetBoundExpr {{.+}} upper
// CHECK:       |     |   | `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       |     |   |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       |     |   `-OpaqueValueExpr [[ove_10]] {{.*}} 'int *__single'
// CHECK:       |     `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |       |-IntegerLiteral {{.+}} 0
// CHECK:       |       `-OpaqueValueExpr [[ove_11]] {{.*}} 'int'
// CHECK:       |-OpaqueValueExpr [[ove_10]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p_5]]
// CHECK:       `-OpaqueValueExpr [[ove_11]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_count_4]]
int *__counted_by_or_null(count) cbn_in_from_single(int count, int *__single p) {
  return p;
}

// CHECK: FunctionDecl [[func_eb_in_from_single:0x[^ ]+]] {{.+}} eb_in_from_single
// CHECK: |-ParmVarDecl [[var_end:0x[^ ]+]]
// CHECK: |-ParmVarDecl [[var_p_6:0x[^ ]+]]
// CHECK: `-CompoundStmt
// CHECK:   `-ReturnStmt
// CHECK:     `-BoundsCheckExpr {{.+}} 'end <= __builtin_get_pointer_upper_bound(p) && p <= end'
// CHECK:       |-ImplicitCastExpr {{.+}} 'int *__single __ended_by(end)':'int *__single' <BoundsSafetyPointerCast>
// CHECK:       | `-OpaqueValueExpr [[ove_12:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:       |-BinaryOperator {{.+}} 'int' '&&'
// CHECK:       | |-BinaryOperator {{.+}} 'int' '<='
// CHECK:       | | |-OpaqueValueExpr [[ove_13:0x[^ ]+]] {{.*}} 'int *__single'
// CHECK:       | | `-GetBoundExpr {{.+}} upper
// CHECK:       | |   `-ImplicitCastExpr {{.+}} 'int *__bidi_indexable' <BoundsSafetyPointerCast>
// CHECK:       | |     `-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single'
// CHECK:       | `-BinaryOperator {{.+}} 'int' '<='
// CHECK:       |   |-OpaqueValueExpr [[ove_12]] {{.*}} 'int *__single'
// CHECK:       |   `-OpaqueValueExpr [[ove_13]] {{.*}} 'int *__single'
// CHECK:       |-OpaqueValueExpr [[ove_12]]
// CHECK:       | `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:       |   `-DeclRefExpr {{.+}} [[var_p_6]]
// CHECK:       `-OpaqueValueExpr [[ove_13]]
// CHECK:         `-ImplicitCastExpr {{.+}} 'int *__single' <LValueToRValue>
// CHECK:           `-DeclRefExpr {{.+}} [[var_end]]
int *__ended_by(end) eb_in_from_single(int *__single end, int *__single p) {
  return p;
}
